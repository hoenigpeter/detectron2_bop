import os
import torch
import copy
import argparse

import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def train(dataset="tless", texture="", num_classes=30):
    register_coco_instances(dataset + texture + "_pbr_meta", {}, "./datasets/BOP_DATASETS/" + dataset + texture + "/train_pbr/000000/scene_gt_coco.json", "./datasets/BOP_DATASETS/" + dataset + texture + "/train_pbr/000000")

    def bop_scenes_to_coco():
        scene_ids = [*range(0, 50, 1)]
        print(scene_ids)
        scene_list = []
        
        for id in scene_ids:
            scene_list.extend(load_coco_json("./datasets/BOP_DATASETS/" + dataset + texture + "/train_pbr/{:06d}/scene_gt_coco.json".format(id), "./datasets/BOP_DATASETS/" + dataset + texture + "/train_pbr/{:06d}".format(id), dataset + texture + "_pbr_meta", None))
        
        return scene_list

    DatasetCatalog.register(dataset + texture + "_pbr_train", bop_scenes_to_coco)

    print("dataset catalog: ", DatasetCatalog.list())
    dataset_temp = DatasetCatalog.get(dataset + texture + "_pbr_train")
    print(MetadataCatalog.get(dataset + texture + "_pbr_train"))

    # Create a Detectron2 config
    # Add a directory to save the model checkpoints
    output_dir = "./frcnn_" + dataset + texture + "_model"
    os.makedirs(output_dir, exist_ok=True)

    # Create a Detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = (dataset + texture + "_pbr_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4  # Adjust according to your system setup
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Pretrained weights
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025

    epochs = 30 

    single_iteration = 1 * cfg.SOLVER.IMS_PER_BATCH
    iterations_for_one_epoch = iterations_for_one_epoch = 50000 / single_iteration

    cfg.SOLVER.MAX_ITER = int(iterations_for_one_epoch * epochs)  # Adjust according to your requirements
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Adjust according to your dataset

    # Set the checkpoint saving options
    cfg.OUTPUT_DIR = output_dir  # Directory to save the checkpoints
    cfg.SOLVER.CHECKPOINT_PERIOD = 20000  # Save a checkpoint every 100 iterations

    # Create the trainer and start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def test(dataset="tless", texture="", num_classes=30):
    register_coco_instances(dataset + "_meta", {}, "./datasets/BOP_DATASETS/" + dataset + "/test/000002/scene_gt_coco.json", "./datasets/BOP_DATASETS/" + dataset + "/test/000002")

    def bop_scenes_to_coco():
        scene_ids = [*range(0, 50, 1)]
        print(scene_ids)
        scene_list = []
        
        for id in scene_ids:
            scene_list.extend(load_coco_json("./datasets/BOP_DATASETS/" + dataset + "/test/{:06d}/scene_gt_coco.json".format(id), "./datasets/BOP_DATASETS/" + dataset + "/test/{:06d}".format(id), dataset + "__meta", None))
        
        return scene_list

    dataset_name = dataset + "_bop_test"

    DatasetCatalog.register(dataset + "_bop_test", bop_scenes_to_coco)

    print("dataset catalog: ", DatasetCatalog.list())

    dataset_temp = DatasetCatalog.get(dataset + "_bop_test")

    print(MetadataCatalog.get(dataset + "_bop_test"))

    # Create a Detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Create a Detectron2 config
    # Add a directory to save the model checkpoints
    output_dir = "./frcnn_" + dataset + "_" + texture + "_model"
    model_path = output_dir + "/model_final.pth"
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # Set the desired threshold for detection
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Adjust according to your dataset
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # Evaluate the model on the lm_bop_test dataset
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(metrics)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='"train" or "test"')
    parser.add_argument('--dataset', type=str, default='tless', help='lm, lmo, tless, itodd')
    parser.add_argument('--texture', type=str, default="", help='original, _random_texture_all or _random_texture')
    parser.add_argument('--num_classes', type=int, default=30, help='number of classes')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    if opt.mode == "train":
        train(opt.dataset, opt.texture, opt.num_classes)
    elif opt.mode == "test":
        test(opt.dataset, opt.texture, opt.num_classes)
    else:
        print("mode is either \"train\" or \"test\"") 

