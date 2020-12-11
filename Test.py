from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.engine.defaults import DefaultPredictor

from detectron2.data.datasets import register_coco_instances

from utils import binary_mask_to_rle

import json
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from pycocotools.mask import decode


def show_rle(rel):
    m = decode(rel)
    print(m.shape)
    plt.imshow(m)
    plt.show()


DATA_ROOT = '/home/ccy-gpl/Datasets/HW3'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
register_coco_instances('tiny-pascal', {}, DATA_ROOT + '/pascal_train.json', DATA_ROOT + '/train_images/')
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 20

DATA_ROOT = '/home/ccy-gpl/Datasets/HW3'
cfg.DATASETS.TRAIN = ('tiny-pascal', )
MetadataCatalog.get("tiny-pascal").thing_classes = ["aeroplane", "bicycle", 'bird', 'boat', 'bottle', 'bus', 'car',
                                                    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                                                    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
cfg.MODEL.WEIGHTS = os.path.join("modules/model_0199999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
# print(predictor.model)
with open(DATA_ROOT + '/test.json') as f:
    test_files = json.load(f)


coco_dt = []
for img_data in tqdm(test_files['images']):
    img = cv2.imread(DATA_ROOT + '/test_images/' + img_data['file_name'])
    h, w, _ = img.shape
    img = cv2.resize(img, (int(w), int(h)))
    outputs = predictor(img)
    '''
    # display
    print(outputs)
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    cv2.imshow('a', out.get_image()[:, :, ::-1])
    k = cv2.waitKey(0)
    if k == ord('q'):
        exit()
    '''
    # Output
    instance = outputs['instances']
    pred_classes = instance.pred_classes
    pred_masks = instance.pred_masks
    pred_scores = instance.scores
    for i in range(len(pred_classes)):
        pred = {}
        pred['image_id'] = img_data['id']
        pred['category_id'] = int(pred_classes[i]+1)
        pred['segmentation'] = binary_mask_to_rle(pred_masks[i, :, :].cpu().numpy())
        pred['score'] = float(pred_scores[i])
        coco_dt.append(pred)

with open("submission.json", "w") as f:
    json.dump(coco_dt, f)
