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

if __name__ == "__main__":
    register_coco_instances("itodd_pbr_train", {}, "datasets/itodd/itodd_annotations_train.json", "datasets/itodd/train_pbr")
    #register_coco_instances("itodd_bop_test_primesense", {}, "datasets/itodd/itodd_annotations_test.json", "datasets/itodd/test_primesense")
    print("dataset catalog: ", DatasetCatalog.list())

    # Create a Detectron2 config
    # Add a directory to save the model checkpoints
    output_dir = "./retinanet_itodd_model"
    os.makedirs(output_dir, exist_ok=True)

    # Create a Detectron2 config for RetinaNet
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("itodd_pbr_train",)
    #cfg.DATASETS.TEST = ("itodd_bop_test_primesense",)
    cfg.DATALOADER.NUM_WORKERS = 4  # Adjust according to your system setup
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Pretrained weights
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025

    epochs = 30 

    single_iteration = 1 * cfg.SOLVER.IMS_PER_BATCH
    iterations_for_one_epoch = 50000 / single_iteration

    cfg.SOLVER.MAX_ITER = int(iterations_for_one_epoch * epochs)  # Adjust according to your requirements
    cfg.MODEL.RETINANET.NUM_CLASSES = 28  # Adjust according to your dataset

    # Set the checkpoint saving options
    cfg.OUTPUT_DIR = output_dir  # Directory to save the checkpoints
    cfg.SOLVER.CHECKPOINT_PERIOD = 30000  # Save a checkpoint every 100 iterations

    # Create the trainer and start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
