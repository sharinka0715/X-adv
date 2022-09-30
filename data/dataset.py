"""
OPIXray Dataset Classes
"""
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
from .rcnn_dataset import RCNNTransform


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, model_classes, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(model_classes, range(len(model_classes))))
        self.model_classes = model_classes
        self.keep_difficult = keep_difficult
        self.type_dict = {}
        self.type_sum_dict = {}

    def __call__(self, target, width, height, idx):
        """
        Arguments:
            target (annotation) : the target annotation
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        with open(target, "r", encoding='utf-8') as f1:
            dataread = f1.readlines()
        for annotation in dataread:
            bndbox = []
            temp = annotation.split()
            name = temp[1]

            if name not in self.model_classes:
                continue
            xmin = int(temp[2]) / width
            if xmin > 1:
                continue
            if xmin < 0:
                xmin = 0
            ymin = int(temp[3]) / height
            if ymin < 0:
                ymin = 0
            xmax = int(temp[4]) / width
            if xmax > 1:           
                xmax = 1
            ymax = int(temp[5]) / height
            if ymax > 1:
                ymax = 1
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)

            label_idx = self.class_to_ind[name]
            # label_idx = name
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        if len(res) == 0:
            return [[0, 0, 0, 0, len(self.model_classes)]]
        return res


class RCNNAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, model_classes, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(model_classes, range(len(model_classes))))
        self.model_classes = model_classes
        self.keep_difficult = keep_difficult
        self.type_dict = {}
        self.type_sum_dict = {}

    def __call__(self, target, width, height, idx):
        """
        Arguments:
            target (annotation) : the target annotation
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        with open(target, "r", encoding='utf-8') as f1:
            dataread = f1.readlines()
        for annotation in dataread:
            bndbox = []
            temp = annotation.split()
            name = temp[1]

            if name not in self.model_classes:
                continue
            xmin = int(temp[2])
            if int(xmin) > width:
                continue
            if xmin < 0:
                xmin = 1
            ymin = int(temp[3])
            if ymin < 0:
                ymin = 1
            xmax = int(temp[4])
            if xmax > width:
                xmax = width
            ymax = int(temp[5])
            if ymax > height:
                ymax = height
            bndbox.append(float(xmin) - 1)
            bndbox.append(float(ymin) - 1)
            bndbox.append(float(xmax) - 1)
            bndbox.append(float(ymax) - 1)

            label_idx = self.class_to_ind[name]
            # label_idx = name
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        if len(res) == 0:
            return [[0, 0, 0, 0, -1]]
        return res


class DetectionDataset(data.Dataset):
    def __init__(self, root,
                 image_sets,
                 model_classes,
                 transform=None, 
                 target_transform=None, 
                 phase=None):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        if target_transform is None:
            self.target_transform = AnnotationTransform(model_classes)
        else:
            self.target_transform = target_transform
        if(phase == 'test'):
            self._annopath = osp.join('%s' % self.root, 'test_annotation', '%s.txt')
            self._imgpath = osp.join('%s' % self.root, 'test_image', '%s.TIFF')
            self._imgpath1 = osp.join('%s' % self.root, 'test_image', '%s.png')
            self._imgpath2 = osp.join('%s' % self.root, 'test_image', '%s.jpg')
        elif(phase == 'train'):
            self._annopath = osp.join('%s' % self.root, 'train_annotation', '%s.txt')
            self._imgpath = osp.join('%s' % self.root, 'train_image', '%s.TIFF')
            self._imgpath1 = osp.join('%s' % self.root, 'train_image', '%s.png')
            self._imgpath2 = osp.join('%s' % self.root, 'train_image', '%s.jpg')
        else:
            print('Directly set', phase, "as the image path")
            self._annopath = osp.join('%s' % self.root, 'test_annotation', '%s.txt')
            self._imgpath = osp.join(phase, '%s.TIFF')
            self._imgpath1 = osp.join(phase, '%s.png')
            self._imgpath2 = osp.join(phase, '%s.jpg')
        self.ids = list()

        with open(self.image_set, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.ids.append(line.strip('\n'))
            

    def __getitem__(self, index):
        im, gt, h, w, og_im, img_id = self.pull_item(index)

        return im, gt, img_id, og_im

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = self._annopath % img_id
        if osp.exists(self._imgpath % img_id):
            img = cv2.imread(self._imgpath % img_id)
        elif osp.exists(self._imgpath1 % img_id):
            img = cv2.imread(self._imgpath1 % img_id)
        elif osp.exists(self._imgpath2 % img_id):
            img = cv2.imread(self._imgpath2 % img_id)
        else:
            print('Error in reading image', img_id)

        try:
            height, width, channels = img.shape
        except:
            print(img_id)
        og_img = img
        try:
            img = cv2.resize(img,(300, 300))
        except:
            print('Error in reading image', img_id)

        if self.target_transform is not None:
            target = self.target_transform(target, width, height, img_id)
            
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, og_img, img_id
        

class RCNNDetectionDataset(data.Dataset):
    def __init__(self, root,
                 image_sets,
                 model_classes,
                 transform=None, 
                 target_transform=None, 
                 phase=None):
        self._dataset = DetectionDataset(root, image_sets, model_classes, transform, target_transform, phase)
        self.transform = RCNNTransform(300, 500)
        self.model_classes = model_classes

    def __getitem__(self, index):
        im, gt, h, w, og_im, img_id = self._dataset.pull_item(index)
        # img, bbox, label, scale = self.transform((im.numpy(), gt))
        im_chw = np.transpose(og_im, (2,0,1)).astype(np.float32)
        img, bbox, label, scale = self.transform((im_chw, gt))
        return torch.from_numpy(img.copy()), bbox, label, scale, img_id

    def __len__(self):
        return len(self._dataset)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    ids = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        ids.append(sample[2])
    return torch.stack(imgs, 0), targets, ids


def detection_collate_attack(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    ids = []
    og_imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        ids.append(sample[2])
        og_imgs.append(sample[3])
    return torch.stack(imgs, 0), targets, ids, og_imgs


def rcnn_detection_collate_attack(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    imgs = []
    bboxes = []
    labels = []
    scales = []
    ids = []
    for sample in batch:
        imgs.append(sample[0].float())
        bboxes.append(torch.FloatTensor(sample[1]))
        labels.append(torch.LongTensor(sample[2]))
        scales.append(sample[3])
        ids.append(sample[4])

    return imgs, bboxes, labels, scales, ids


def detection_collate_test(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    ids = []
    heights = []
    widths = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        ids.append(sample[2])
        heights.append(sample[3].shape[0])
        widths.append(sample[3].shape[1])
    return torch.stack(imgs, 0), targets, ids, heights, widths


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
