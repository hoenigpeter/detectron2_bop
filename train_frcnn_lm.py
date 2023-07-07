import os
import torch
import copy
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.model_zoo import model_zoo

from configs.tless_pbr import register_with_name_cfg

if __name__ == "__main__":
    from configs.lm_pbr import register_with_name_cfg
    register_with_name_cfg("lm_pbr_train")
    from configs.lm_bop_test import register_with_name_cfg
    register_with_name_cfg("lm_bop_test")

    print("dataset catalog: ", DatasetCatalog.list())

    # Create a Detectron2 config
    # Add a directory to save the model checkpoints
    output_dir = "./frcnn_lm_model"
    os.makedirs(output_dir, exist_ok=True)

    # Create a Detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("lm_pbr_train",)
    cfg.DATASETS.TEST = ("lm_bop_test",)
    cfg.DATALOADER.NUM_WORKERS = 4  # Adjust according to your system setup
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Pretrained weights
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025

    epochs = 1 

    single_iteration = 1 * cfg.SOLVER.IMS_PER_BATCH
    iterations_for_one_epoch = iterations_for_one_epoch = 50000 / single_iteration

    cfg.SOLVER.MAX_ITER = int(iterations_for_one_epoch * epochs)  # Adjust according to your requirements
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15  # Adjust according to your dataset

    # Set the checkpoint saving options
    cfg.OUTPUT_DIR = output_dir  # Directory to save the checkpoints
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save a checkpoint every 100 iterations

    # Create the trainer and start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

