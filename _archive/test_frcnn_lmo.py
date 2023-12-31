import os
import torch
import copy
import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from configs.tless_pbr import register_with_name_cfg

if __name__ == "__main__":
    from configs.lmo_bop_test import register_with_name_cfg
    register_with_name_cfg("lmo_bop_test")

    print("dataset catalog: ", DatasetCatalog.list())

    # Create a Detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Create a Detectron2 config
    # Add a directory to save the model checkpoints
    output_dir = "./frcnn_lmo_random_texture_all_model_with_aug"
    model_path = output_dir + "/model_final.pth"
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # Set the desired threshold for detection
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Adjust according to your dataset

    metadata = MetadataCatalog.get("lmo_bop_test")
    dataset_dicts = DatasetCatalog.get("lmo_bop_test")

    predictor = DefaultPredictor(cfg)
    # Evaluate the model on the lm_bop_test dataset
    evaluator = COCOEvaluator("lmo_bop_test", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "lmo_bop_test")
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(metrics)

