import os
import time
import random
import argparse
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.utils.data as data
import numpy as np

from data.dataset import RCNNDetectionDataset, RCNNAnnotationTransform
from data import config
from model import FasterRCNNVGG16, FasterRCNNTrainer

# os.environ["NCCL_DEBUG"] = "INFO"

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for the experiments')
parser.add_argument('--dataset', default="OPIXray", type=str, 
                    choices=["OPIXray", "HiXray", "XAD"], help='Dataset name')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--transfer', default=None, type=str,
                    help='Checkpoint state_dict file to transfer from')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1*(1e-3), type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default=None, type=str, required=True,
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

fix_seed(args.seed)

torch.set_default_tensor_type('torch.FloatTensor')

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")

start_time = time.strftime ('%Y-%m-%d_%H-%M-%S')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder, exist_ok=True)

def train():
    print(f'Training for {args.dataset} model...')  

    if args.dataset == "OPIXray":
        data_info = config.OPIXray_train
    elif args.dataset == "HiXray":
        data_info = config.HiXray_train
    elif args.dataset == "XAD":
        data_info = config.XAD_train
    cfg = config.FasterRCNN

    # modify cfg
    cfg['lr'] = args.lr
    cfg['weight_decay'] = args.weight_decay
    num_classes = len(data_info["model_classes"]) + 1

    dataset = RCNNDetectionDataset(root=data_info["dataset_root"], 
                               model_classes=data_info["model_classes"],
                               image_sets=data_info["imagesetfile"], 
                               target_transform=RCNNAnnotationTransform(data_info["model_classes"]), 
                               phase='train')

    frcnn_net = FasterRCNNVGG16(cfg, num_classes - 1, transfer=args.transfer)
    trainer = FasterRCNNTrainer(frcnn_net, cfg, num_classes).cuda()

    print(frcnn_net)

    trainer.train()
    # loss counters
    epoch = 0
    print('Loading dataset...')

    print('Training SSD on', args.dataset)
    print('Using the specified args:')
    print(args)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True)

    for epoch in range(200):
        trainer.reset_meters()
        loss_cnt = 0
        for ii, (img, bbox_, label_, scale, ids) in enumerate(data_loader):
            # print(img.shape, bbox_, label_, scale, ids)
            scale = scale.item()
            # print(ids)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            losses = trainer.train_step(img, bbox, label, scale)
            loss_cnt += losses.total_loss.item()
            if (ii + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, iter {ii+1}/{len(data_loader)}, losses: {loss_cnt / 50}")
                loss_cnt = 0

        # if (epoch + 1) % 1 == 0:
        torch.save(frcnn_net.state_dict(), args.save_folder + '/model_' +
                    repr(epoch+1) + '.pth')

train()