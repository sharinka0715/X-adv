import os
import random
import argparse
import numpy as np
import warnings
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from model import FasterRCNNVGG16

from data.dataset import RCNNDetectionDataset, RCNNAnnotationTransform
from data import config

warnings.filterwarnings("ignore")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument("--ckpt_path", default="../weights/DOAM/OPIX.pth", type=str, 
                    help="the checkpoint path of the model")

parser.add_argument('--dataset', default="OPIXray", type=str, 
                    choices=["OPIXray", "HiXray", "XAD"], help='Dataset name')
parser.add_argument('--phase', default='test', type=str,
                    help='test phase')

parser.add_argument('--batch_size', default=1, type=int,
                    help='The size of a mini batch')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')


args = parser.parse_args()

fix_seed(0)

torch.set_default_tensor_type('torch.FloatTensor')

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
    
    
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    

def cal_ap(boxes, gts, npos, name, ovthresh=0.5):
    full_boxes = []
    for elm in boxes:
        if len(elm) > 0:
            full_boxes.append(elm)
    if len(full_boxes) == 0:
        return 0.0
    boxes = np.concatenate(full_boxes, 0) # [num_images*num_boxes, 5]
    
    # sort by confidence
    confidence = boxes[:, 4]
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    image_ids = [int(boxes[x, 5]) for x in sorted_ind]
    BB = boxes[sorted_ind, :]
    
    # mark TPs and FPs
    nd = len(image_ids)

    tp = np.zeros(nd)
    fp = np.zeros(nd)

    avg_conf = 0
    tp_num = 0
    
    count = 0

    for d in range(nd):
        gt = np.array(gts[image_ids[d]])
        if len(gt) == 0:
            fp[d] = 1.
            count += 1
            continue
        bb = BB[d, :4].astype(float)
        ovmax = -np.inf
        BBGT = gt[:, :4].astype(float)

        # compute overlaps
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
               (BBGT[:, 2] - BBGT[:, 0]) *
               (BBGT[:, 3] - BBGT[:, 1]) - inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if gts[image_ids[d]][jmax][5] == 0:
                tp[d] = 1.
                gts[image_ids[d]][jmax][5] = 1
                avg_conf += BB[d, 4]
                tp_num += 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    
    rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec, False)
    return ap


def cal_map(all_boxes, all_gts, labelmap):
    """
    calculate mAP as PASCAL VOC 2010
    params:
        all_boxes: 
            needs shape as (num_classes, num_images, num_boxes, 6)
            6 means [x1, y1, x2, y2, conf, img_id]
        all_gts:
            needs shape as (num_classes, num_images, num_boxes, 6)
            6 means [x1, y1, x2, y2, label, is_chosen(default to be 0)]
    """
    mAP = 0
    total = 0
    print("labelmap:{}".format(labelmap))
    for i, cls in enumerate(labelmap):
        npos = 0
        for elm in all_gts[i]:
            npos += len(elm)
        if npos == 0:
            continue
        ap = cal_ap(all_boxes[i], all_gts[i], npos, cls)
        print("AP for {}: {:.4f}".format(cls, ap))
        if not np.isnan(ap):
            mAP += ap
            total += 1
    print("mAP: {:.4f}".format(mAP / total))


def test_net(net, cuda, dataset, labelmap, im_size=300):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap))]
    
    all_gts = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap))]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    nob = 0
    avg_conf = 0
    for i, (images, bboxes, labels, scale, img_id) in enumerate(tqdm(loader)):
        x = images.type(torch.cuda.FloatTensor)
        for j in range(images.shape[0]):
            for k in range(bboxes[j].shape[0]):
                bbox = bboxes[j][k].numpy().tolist()
                label = labels[j][k].item()
                t = i * args.batch_size + j
                all_gts[label][t].append([bbox[0], bbox[1], bbox[2], bbox[3], label, 0])

        with torch.no_grad():
            boxes_, labels_, scores_ = net.predict(x, sizes=[images[zz].shape[1:] for zz in range(images.shape[0])])
        for k in range(len(labels_)):
            for m in range(len(labels_[k])):
                boxes = boxes_[k][m]
                labels = labels_[k][m]
                scores = scores_[k][m]

                img_ids = i * args.batch_size + k
                cls_dets = torch.tensor([boxes[0], boxes[1], boxes[2], boxes[3], scores, img_ids])
                all_boxes[labels][i*args.batch_size+k].append(cls_dets.cpu().numpy())
                nob += 1
                avg_conf += scores
    
    confs = []
    for elm_cls in all_boxes:
        for elm in elm_cls:
            for e in elm:
                confs.append(e[4])
    cal_map(all_boxes, all_gts, labelmap)


if __name__ == '__main__':
    print(args)
    if args.dataset == "OPIXray":
        data_info = config.OPIXray_test
    elif args.dataset == "HiXray":
        data_info = config.HiXray_test
    elif args.dataset == "XAD":
        data_info = config.XAD_test

    num_classes = len(data_info["model_classes"]) + 1

    net = FasterRCNNVGG16(config.FasterRCNN, num_classes - 1)

    net.load_state_dict(torch.load(args.ckpt_path))
    net.eval()
    
    dataset = RCNNDetectionDataset(root=data_info["dataset_root"], 
                            model_classes=data_info["model_classes"],
                            image_sets=data_info["imagesetfile"], 
                            target_transform=RCNNAnnotationTransform(data_info["model_classes"]), 
                            phase=args.phase)

    if args.cuda:
        net = net.cuda()
        
    test_net(net, args.cuda, dataset, data_info["model_classes"], 300)
    print(args.ckpt_path, args.phase)

