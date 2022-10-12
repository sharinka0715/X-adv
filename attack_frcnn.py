"""
Xray Adversarial Attack
"""
import os
import sys
import cv2
import math
import time
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from model import FasterRCNNVGG16, FasterRCNNTrainer

from data.dataset import RCNNDetectionDataset, rcnn_detection_collate_attack, RCNNAnnotationTransform
from data import config
from utils import stick, renderer, rl_utils

warnings.filterwarnings("ignore")
torch.set_default_tensor_type('torch.FloatTensor')

parser = argparse.ArgumentParser(description="X-ray adversarial attack.")
# for model
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for the experiments')
parser.add_argument("--ckpt_path", default="./ckpt/OPIX.pth", type=str, 
                    help="the checkpoint path of the model")
# for data
parser.add_argument('--dataset', default="OPIXray", type=str, 
                    choices=["OPIXray", "HiXray"], help='Dataset name')
parser.add_argument("--phase", default="test", type=str, 
                    help="the phase of the X-ray image dataset")
parser.add_argument("--batch_size", default=10, type=int, 
                    help="the batch size of the data loader")
parser.add_argument("--num_workers", default=4, type=int, 
                    help="the number of workers of the data loader")
# for patch
parser.add_argument("--obj_path", default="objs/ball_small.obj", type=str, 
                    help="the path of adversarial 3d object file")
parser.add_argument("--patch_size", default=20, type=int, 
                    help="the size of X-ray patch")
parser.add_argument("--patch_count", default=4, type=int, 
                    help="the number of X-ray patch")
parser.add_argument("--patch_place", default="reinforce", type=str, choices=['none', 'fix', 'fix_patch', 'reinforce'],
                    help="the place where the X-ray patch located")
parser.add_argument("--patch_material", default="iron", type=str, choices=["iron", "plastic", "aluminum", "iron_fix"],
                    help="the material of patch, which decides the color of patch")            
# for attack
parser.add_argument("--targeted", default=False, action="store_true",
                    help="whether to use targeted (background) attack")
parser.add_argument("--lr", default=0.01, type=float, 
                    help="the learning rate of attack")
parser.add_argument("--beta", default=0.01, type=float, 
                    help="the perceptual loss rate of attack")
parser.add_argument("--num_iters", default=24, type=int, 
                    help="the number of iterations of attack")
parser.add_argument("--save_path", default="../results", type=str,
                    help="the save path of adversarial examples")

timer = time.time()
def stime(content):
    global timer
    torch.cuda.synchronize()
    print(content, time.time() - timer)
    timer = time.time()

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

args = parser.parse_args()
args.save_path = os.path.join(args.save_path, f"{args.dataset}/{args.patch_material}/{args.patch_place}", "FasterRCNN")

fix_seed(args.seed)
print(args)

    
def get_place_fix(images, bboxes):
    fix_place_list = ["nw", "ne", "sw", "se", "n", "s", "w", "e"]
    areas_choose = [[] for _ in range(len(images))]
    for i in range(args.patch_count):
        places = stick.cal_stick_place_rcnn(bboxes, args.patch_size, args.patch_size, 0.25, fix_place_list[i])
        for j in range(len(places)):
            areas_choose[j].append(places[j])
            
    return areas_choose

def get_place_reinforce(images, bboxes, group, faces, trainer):
    actor = rl_utils.Actor(args.patch_count).cuda()
    actor.train()
    actor_optim = optim.Adam(actor.parameters(), lr=9e-4)

    areas = stick.get_stick_area_rcnn(bboxes, args.patch_size, args.patch_size)
    area_lens = torch.FloatTensor([len(elm) for elm in areas]).unsqueeze(1)

    pad = nn.ZeroPad2d(args.patch_size)
    group_clamp = torch.clamp(group, 0, 1)

    # use X-ray renderer to convert a 3D object to an X-ray image
    rend_group = []
    for pt in range(args.patch_count):
        depth_img = renderer.ball2depth(group_clamp[pt], faces, args.patch_size, args.patch_size).unsqueeze(0).unsqueeze(0)
        # simulate function needs a 4-dimension input
        rend, mask = renderer.simulate(depth_img, args.patch_material)
        rend[~mask] = 1
        rend_group.append(rend)

    # Using running mean/std to stablize the reward
    reward_ms = rl_utils.RunningMeanStd(shape=(1,), device="cuda:0")

    last_reward = 0
    ori_shape = [e.shape for e in images]

    for rep in range(200):
        print("RL phase", rep + 1)
        # sample actions
        action_logits = []
        for s in range(len(images)):
            # resize(images[s], (224, 224))
            action_logits.append(actor(images[s].unsqueeze(0)))
        action_logits = torch.cat(action_logits, dim=0)
        dist = Categorical(logits=action_logits)
        actions = dist.sample()

        places = (area_lens * actions.detach().cpu() / 50).floor().long()
        areas_choose = []
        for bi in range(places.shape[0]):
            area_choose = []
            for pi in range(places.shape[1]):
                area_choose.append(areas[bi][places[bi, pi]])
            areas_choose.append(area_choose)

        # get rewards
        images_delta = [e.clone().detach() for e in images]
        images_delta = [pad(e) for e in images_delta]
        for pt in range(args.patch_count):
            rend = rend_group[pt]
            for s in range(len(images_delta)):
                u, v = areas_choose[s][pt]
                images_delta[s][:, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend[0])

        images_cutpad = []
        for s in range(len(images_delta)):
            images_cutpad.append(images_delta[s][:, args.patch_size:ori_shape[s][1]+args.patch_size, args.patch_size:ori_shape[s][2]+args.patch_size])

        rewards = []
        with torch.no_grad():
            for s in range(len(images_cutpad)):
                trainer.reset_meters()
                loss_cls = trainer.forward(images_cutpad[s].unsqueeze(0), bboxes[s].unsqueeze(0), labels[s].unsqueeze(0), scales[s])
                rewards.append(loss_cls.rpn_cls_loss + loss_cls.roi_cls_loss)
        rewards = torch.stack(rewards, dim=0).unsqueeze(1).detach()

        if actions.shape[-1] > 1:
            reward_penal = actions.float().std(dim=-1, keepdim=True)
        else:
            reward_penal = torch.tensor([0.]).cuda()
        print(rewards.mean().item(), reward_penal.mean().item())
        rewards += 0.01 * reward_penal

        # early stopping
        cur_reward = rewards.mean().item()
        if cur_reward == last_reward:
            break
        last_reward = cur_reward
            
        # standarize rewards
        reward_ms.update(rewards)
        rewards = (rewards - reward_ms.mean) / torch.sqrt(reward_ms.var)

        # learn
        log_prob = dist.log_prob(actions)
        loss = -(rewards * log_prob).mean()

        actor_optim.zero_grad()
        loss.backward()
        actor_optim.step()

        actions = actions.float()

    return areas_choose


def attack(images, bboxes, labels, net, trainer):
    """
    Main attack function.
    """
    net.phase = "train"
    images = [e.cuda() for e in images]
    bboxes = [b.cuda() for b in bboxes]
    labels = [l.cuda() for l in labels]

    # create a group of patch objects which have same faces
    # we only optimize the coordinate of vertices
    # but not to change the adjacent relation
    group = []
    for _ in range(args.patch_count):
        vertices, faces = renderer.load_from_file(args.obj_path)
        group.append(vertices.unsqueeze(0))

    adj_ls = renderer.adj_list(vertices, faces)
    
    # the shape of group: [patch_count, 3, vertices_count]
    group = torch.cat(group, dim=0).cuda()
    group_ori = group.clone().detach()
    depth_patch = torch.zeros((1, args.patch_count, args.patch_size, args.patch_size)).uniform_().cuda()

    if not args.patch_place == "fix_patch":
        group.requires_grad_(True)
        optimizer = optim.Adam([group], lr=args.lr)
    else:
        depth_patch.requires_grad_(True)
        optimizer = optim.Adam([depth_patch], lr=args.lr)
    # we need a pad function to prevent that a part of patch is out of the image
    pad = nn.ZeroPad2d(args.patch_size)
    
    print("Calculate best place before attack...")
    if args.patch_place == "fix" or args.patch_place == "fix_patch":
        areas_choose = get_place_fix(images, bboxes)
    elif args.patch_place == "reinforce":
        areas_choose = get_place_reinforce(images, bboxes, group, faces, trainer)

    print("Attacking...")
    ori_shape = [e.shape for e in images]
    for t in range(args.num_iters):
        timer = time.time()
        
        images_delta = [e.clone().detach() for e in images]
        images_delta = [pad(e) for e in images_delta]
        
        # calculate the perspective loss
        loss_per = torch.zeros((1,)).cuda()
        if not args.patch_place == "fix_patch":
            for pt in range(args.patch_count):
                loss_per += renderer.tvloss(group_ori[pt], group[pt], adj_ls, coe=0)
            loss_per /= args.patch_count
        
        # clamp the group into [0, 1]
        group_clamp = torch.clamp(group, 0, 1)
        depth_clamp = torch.clamp(depth_patch, 0, 1)
        
        # use X-ray renderer to convert a 3D object to an X-ray image
        for pt in range(args.patch_count):
            if not args.patch_place == "fix_patch":
                depth_img = renderer.ball2depth(group_clamp[pt], faces, args.patch_size, args.patch_size).unsqueeze(0).unsqueeze(0)
            else:
                depth_img = depth_clamp[:, pt:pt+1]
            # simulate function needs a 4-dimension input
            rend, mask = renderer.simulate(depth_img, args.patch_material.replace("_fix", ""))
            rend[~mask] = 1
            for s in range(len(images_delta)):
                u, v = areas_choose[s][pt]
                # print(rend.shape, images_delta[s].shape)
                # print(u, v, bboxes[s])
                images_delta[s][:, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend[0])
        images_cutpad = []
        for s in range(len(images_delta)):
            images_cutpad.append(images_delta[s][:, args.patch_size:ori_shape[s][1]+args.patch_size, args.patch_size:ori_shape[s][2]+args.patch_size])
        # out = net(images_delta[:, :, args.patch_size:300+args.patch_size, args.patch_size:300+args.patch_size])
        loss = 0
        for s in range(len(images_cutpad)):
            trainer.reset_meters()
            loss_cls = trainer.forward(images_cutpad[s].unsqueeze(0), bboxes[s].unsqueeze(0), labels[s].unsqueeze(0), scales[s])
            loss += loss_cls.rpn_cls_loss + loss_cls.roi_cls_loss
        loss /= len(images_cutpad)
        
        loss_adv = - loss
        loss_total = loss_adv + args.beta * loss_per

        optimizer.zero_grad()
        loss_total.backward()
        if not args.patch_place == "fix_patch":
            inan = group.grad.isnan()
            group.grad.data[inan] = 0
        optimizer.step()
        
        torch.cuda.synchronize()
        
        print("Iter: {}/{}, L_adv = {:.3f}, βL_per = {:.3f}, Total loss = {:.3f}, Time: {:.2f}".format(
            t+1, args.num_iters, loss_adv.item() * 1000, args.beta * loss_per.item() * 1000,
            loss_total.item() * 1000, time.time() - timer))
            
    print("Calculate best place after attack...")

    group_clamp = torch.clamp(group, 0, 1)
    depth_clamp = torch.clamp(depth_patch, 0, 1)
    images_adv = [pad(e).clone().detach() for e in images]
    for pt in range(args.patch_count):
        if not args.patch_place == "fix_patch":
            depth_img = renderer.ball2depth(group_clamp[pt], faces, args.patch_size, args.patch_size).unsqueeze(0).unsqueeze(0)
        else:
            depth_img = depth_clamp[:, pt:pt+1]
        # simulate function needs a 4-dimension input
        rend, mask = renderer.simulate(depth_img, args.patch_material)
        rend[~mask] = 1
        for s in range(len(images_adv)):
            u, v = areas_choose[s][pt]
            images_adv[s][:, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend[0])

    images_cutpad = []
    for s in range(len(images_adv)):
        images_cutpad.append(images_adv[s][:, args.patch_size:ori_shape[s][1]+args.patch_size, args.patch_size:ori_shape[s][2]+args.patch_size])
        
    return images_cutpad, areas_choose, torch.clamp(group, 0, 1), faces
    

def save_img(path, img_tensor, shape):
    img_tensor = img_tensor.cpu().detach().numpy().astype(np.uint8)
    img = img_tensor.transpose(1, 2, 0)
    img = cv2.resize(img, (shape[1], shape[0]))
    cv2.imwrite(path, img)


if __name__ == "__main__":
    if args.dataset == "OPIXray":
        data_info = config.OPIXray_test
    elif args.dataset == "HiXray":
        data_info = config.HiXray_test

    num_classes = len(data_info["model_classes"]) + 1
    net = FasterRCNNVGG16(config.FasterRCNN, num_classes - 1)
    trainer = FasterRCNNTrainer(net, config.FasterRCNN, num_classes).cuda()
    net.load_state_dict(torch.load(args.ckpt_path))

    print("CUDA is available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("Warning! CUDA is not supported on your device!")
        sys.exit(0)
    else:
        print("CUDA visible device count:", torch.cuda.device_count())
        
    net.eval()

    dataset = RCNNDetectionDataset(root=data_info["dataset_root"], 
                               model_classes=data_info["model_classes"],
                               image_sets=data_info["imagesetfile"], 
                               target_transform=RCNNAnnotationTransform(data_info["model_classes"]), 
                               phase='test')
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=rcnn_detection_collate_attack, pin_memory=True)

    num_images = len(dataset)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    img_path = os.path.join(args.save_path, "adver_image")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
        
    obj_path = os.path.join(args.save_path, "adver_obj")
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)

    for i, (images, bboxes, labels, scales, img_ids) in enumerate(data_loader):
        print("Batch {}/{}...".format(i+1, math.ceil(num_images / args.batch_size)))
        print(img_ids)
        if args.patch_place != "none":
            images_adv, areas_choose, vertices, faces = attack(images, bboxes, labels, net, trainer)
        else:
            images_adv = images
            faces = None
        
        print("Saving...")
        for t in range(len(images_adv)):
            shape = images_adv[t].shape[1:]
            shape = [int(shape[0] / scales[t]), int(shape[1] / scales[t])]
            save_img(os.path.join(img_path, img_ids[t] + ".png"), images_adv[t] * 255, shape)
            if faces is not None:
                for i in range(vertices.shape[0]):
                    renderer.save_to_file(
                        os.path.join(obj_path, str(img_ids[t]) + "_u{}_v{}.obj".format(*areas_choose[t][i])), 
                        vertices[i], faces)