import os
import time
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np

from data.dataset import DetectionDataset, detection_collate, AnnotationTransform
from data import config
from layers.modules import MultiBoxLoss


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
parser.add_argument('--model_arch', default="original", type=str, 
                    choices=["original", "DOAM", "LIM"], help='Model architecture')
parser.add_argument('--batch_size', default=24, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--transfer', default=None, type=str,
                    help='Checkpoint state_dict file to transfer from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1*(1e-4), type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default=None,type=str,
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
    print(f'Training {args.model_arch} model on {args.dataset} dataset...')
    if args.dataset == "OPIXray":
        data_info = config.OPIXray_train
    elif args.dataset == "HiXray":
        data_info = config.HiXray_train
    elif args.dataset == "XAD":
        data_info = config.XAD_train

    num_classes = len(data_info["model_classes"]) + 1

    dataset = DetectionDataset(root=data_info["dataset_root"], 
                               model_classes=data_info["model_classes"],
                               image_sets=data_info["imagesetfile"], 
                               target_transform=AnnotationTransform(data_info["model_classes"]), 
                               phase='train')

    if args.model_arch == "DOAM":
        from model.ssd_doam import build_ssd
        cfg = config.DOAM
        ssd_net = build_ssd("train", size=300, num_classes=num_classes)
    elif args.model_arch == "LIM":
        from model.ssd_lim import build_ssd
        cfg = config.LIM
        ssd_net = build_ssd("train", size=300, num_classes=num_classes)
    elif args.model_arch == "original":
        from model.ssd_original import build_ssd
        cfg = config.original
        ssd_net = build_ssd("train", size=300, num_classes=num_classes)

    net = ssd_net
    print(ssd_net)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)

    if args.model_arch == "DOAM":
        ssd_net._modules['vgg'][0] = nn.Conv2d(4, 64, kernel_size=3, padding=1)

    if args.transfer:
        print('Transfer learning...')
        ssd_net.load_weights(args.transfer, isStrict=False)
        ssd_net.conf_fpn.apply(weights_init)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
    
    if (not args.resume) & (not args.transfer) :
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net._conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, cfg['variance'])

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', args.dataset)
    print('Using the specified args:')
    print(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
            loc_loss = 0
            conf_loss = 0
            if epoch % 5 == 0:
                print('Saving state, epoch:', epoch)
                torch.save(ssd_net.state_dict(), args.save_folder + '/ssd300_Xray_knife_' +
                           repr(epoch) + '.pth')
            

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        print ('iteration:', iteration)

        try:
            images, targets, ids = next(batch_iterator)
        except:
            batch_iterator = iter(data_loader)
            images, targets, ids = next(batch_iterator)
            print ('Reload!')
       

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        images = images.type(torch.FloatTensor)
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    '''Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    '''
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__ == '__main__':
    train()
