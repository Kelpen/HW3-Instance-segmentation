from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.config.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.engine.defaults import DefaultTrainer

from MyMapper import mapper_train_loader
from PatchRCNN import PatchRCNN


def main():
    # Dataset
    register_coco_instances('tiny-pascal', {}, DATA_ROOT + '/pascal_train.json', DATA_ROOT + '/train_images/')

    # Config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ('tiny-pascal',)
    cfg.DATASETS.TEST = ()
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 20

    cfg.MODEL.META_ARCHITECTURE = 'PatchRCNN'
    cfg.MODEL.WEIGHTS = './models/R-50.pkl'
    # model weights: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md

    cfg.INPUT.MAX_SIZE_TRAIN = 500
    cfg.INPUT.MIN_SIZE_TRAIN = (300, 500)

    cfg.OUTPUT_DIR = './models'

    # trainer
    DefaultTrainer.build_train_loader = mapper_train_loader
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load()
    trainer.train()


if __name__ == '__main__':
    DATA_ROOT = '/home/ccy-gpl/Datasets/HW3'
    main()
