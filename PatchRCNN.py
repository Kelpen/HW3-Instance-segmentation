from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.structures import BitMasks, Instances, Boxes
from typing import Dict, Tuple

import torch
from torch.nn.functional import interpolate
import numpy as np


# This is a wrapper
#
@META_ARCH_REGISTRY.register()
class PatchRCNN(GeneralizedRCNN):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg):
        return super(PatchRCNN, cls).from_config(cfg)

    def forward(self, batched_inputs):
        images = []
        masks = []
        classes = []
        boxes = []
        img_sizes = []
        num_instance = [0]
        for data in batched_inputs:
            images.append(data['image'])
            img_sizes.append(images[-1].shape[-2:])
            classes.append(data['instances'].gt_classes)
            boxes.append(data['instances'].gt_boxes.tensor)
            masks.append(data['instances'].gt_masks.tensor)
            num_instance.append(len(classes[-1]))
        num_instance = np.cumsum(num_instance)

        image_arrange = [[[0, 1], [2, 3]], [[2, 3], [0, 1]], [[1, 0], [3, 2]], [[3, 2], [1, 0]]]
        new_data = []
        for i in range(2):
            h, w = images[i].shape[-2:]
            resized_imgs = []
            resized_masks = []
            boxes_processed = []

            for j in range(4):
                img = images[j]
                mask = masks[j]
                box = boxes[j]
                if torch.rand(1) > 0.5:  # flip
                    img = torch.flip(img, dims=(-1,))
                    mask = torch.flip(mask, dims=(-1,))
                    box = (box * torch.tensor([[-1, 1, -1, 1]])) + torch.tensor(
                        [[img_sizes[j][1], 0, img_sizes[j][1], 0]])
                    box = box[:, [2, 1, 0, 3]]
                resized_imgs.append(interpolate(img[None], (h, w))[0])
                resized_masks.append(interpolate(mask[None].type(torch.float), (h, w))[0].type(torch.bool))
                boxes_processed.append(box)
            ab = torch.cat([resized_imgs[image_arrange[i][0][0]], resized_imgs[image_arrange[i][0][1]]], dim=2)
            cd = torch.cat([resized_imgs[image_arrange[i][1][0]], resized_imgs[image_arrange[i][1][1]]], dim=2)
            abcd = torch.cat([ab, cd], dim=1).type(torch.float)
            big_img = abcd + torch.randn_like(abcd) * 8

            big_mask = torch.zeros((num_instance[-1], h * 2, w * 2), dtype=torch.bool)
            (a, b), (c, d) = image_arrange[i]

            all_boxes = torch.zeros((num_instance[-1], 4))
            big_mask[num_instance[a]:num_instance[a + 1], :h, :w] = resized_masks[a]
            all_boxes[num_instance[a]:num_instance[a + 1]] = boxes_processed[a].clone()
            all_boxes[num_instance[a]:num_instance[a + 1]][:, [0, 2]] *= w / img_sizes[a][1]
            all_boxes[num_instance[a]:num_instance[a + 1]][:, [1, 3]] *= h / img_sizes[a][0]
            big_mask[num_instance[b]:num_instance[b + 1], :h, w:] = resized_masks[b]
            all_boxes[num_instance[b]:num_instance[b + 1]] = boxes_processed[b].clone()
            all_boxes[num_instance[b]:num_instance[b + 1]][:, [0, 2]] *= w / img_sizes[b][1]
            all_boxes[num_instance[b]:num_instance[b + 1]][:, [1, 3]] *= h / img_sizes[b][0]
            all_boxes[num_instance[b]:num_instance[b + 1]][:, [0, 2]] += w
            big_mask[num_instance[c]:num_instance[c + 1], h:, :w] = resized_masks[c]
            all_boxes[num_instance[c]:num_instance[c + 1]] = boxes_processed[c].clone()
            all_boxes[num_instance[c]:num_instance[c + 1]][:, [0, 2]] *= w / img_sizes[c][1]
            all_boxes[num_instance[c]:num_instance[c + 1]][:, [1, 3]] *= h / img_sizes[c][0]
            all_boxes[num_instance[c]:num_instance[c + 1]][:, [1, 3]] += h
            big_mask[num_instance[d]:num_instance[d + 1], h:, w:] = resized_masks[d]
            all_boxes[num_instance[d]:num_instance[d + 1]] = boxes_processed[d].clone()
            all_boxes[num_instance[d]:num_instance[d + 1]][:, [0, 2]] *= w / img_sizes[d][1]
            all_boxes[num_instance[d]:num_instance[d + 1]][:, [1, 3]] *= h / img_sizes[d][0]
            all_boxes[num_instance[d]:num_instance[d + 1]][:, [0, 2]] += w
            all_boxes[num_instance[d]:num_instance[d + 1]][:, [1, 3]] += h

            big_masks = BitMasks(big_mask)
            big_boxes = Boxes(all_boxes)
            big_classes = torch.cat(classes, dim=0)
            new_instance = Instances(image_size=(h, w), gt_boxes=big_boxes, gt_masks=big_masks, gt_classes=big_classes)
            data = {'file_name': batched_inputs[i]['file_name'], 'height': h, 'width': w,
                    'image_id': batched_inputs[i]['image_id'],
                    'image': big_img, 'instances': new_instance}
            new_data.append(data)

        loss = super(PatchRCNN, self).forward(new_data)
        return loss
