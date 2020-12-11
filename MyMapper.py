from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures.masks import BitMasks
from detectron2.data.build import build_detection_train_loader
from detectron2.config import configurable
from torchvision.transforms import ColorJitter

import torch
import cv2
import numpy as np


class TrainMapper(DatasetMapper):
    @configurable
    def __init__(self, is_train, **kwargs):
        super(TrainMapper, self).__init__(is_train, **kwargs)
        self.sharpen_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        return super(TrainMapper, cls).from_config(cfg, is_train)

    def __call__(self, dataset_dict):
        # read data
        data = super(TrainMapper, self).__call__(dataset_dict)
        instance = data['instances']

        # decode gt_mask from polygon to bitmask
        h, w = instance.image_size
        instance.gt_masks = BitMasks.from_polygon_masks(instance.gt_masks, h, w)

        # augmentations
        img = data['image'].numpy().transpose(1, 2, 0).astype(np.float)
        sk = self.sharpen_kernel * float((torch.randn(1)-0.5) / 8)
        sk[1, 1] += 1
        img = cv2.filter2D(img, -1, sk)
        point_factor = w*h//2048
        for i in range(point_factor):
            center = [int(xx) for xx in (torch.rand(2) * torch.tensor([w, h]))]
            color = tuple([int(xx) for xx in torch.randint(0, 255, (4,))])
            size = int(torch.randint(2, 5, (1,)))
            img = cv2.circle(img, tuple(center), size, color=color, thickness=-1)
        sk[1, 1] -= 1
        data['image'] = self.color_jitter(torch.Tensor(img.transpose(2, 0, 1)) / 255) * 255

        return data


def mapper_train_loader(cls, cfg):
    return build_detection_train_loader(cfg, mapper=TrainMapper(cfg))
