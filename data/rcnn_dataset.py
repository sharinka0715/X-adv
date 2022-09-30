from __future__ import  absolute_import
from __future__ import  division
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np


def inverse_normalize(img):
    return img.clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0, 0, 0],
                                std=[1, 1, 1])
    img = normalize(t.from_numpy(img))
    return img.numpy()



def preprocess(img, min_size=300, max_size=500):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    normalize = pytorch_normalze
    return normalize(img)


class RCNNTransform(object):
    def __init__(self, min_size=300, max_size=500):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, target = in_data
        bbox = []
        label = []
        for e in target:
            # bbox.append(e[:4])
            # change [x1, y1, x2, y2] into [y1, x1, y2, x2]
            bbox.append([e[1], e[0], e[3], e[2]])
            label.append(e[4])
        bbox = np.array(bbox)
        label = np.array(label)
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        # img, params = util.random_flip(
        #     img, x_random=False, return_param=True)
        # bbox = util.flip_bbox(
        #     bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale
