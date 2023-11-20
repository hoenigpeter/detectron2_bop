import logging
import os
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.model_zoo import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from evaluators.bop_evaluator import BOPEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import argparse

from detectron2.engine import (
    default_writers,
)
import json

import numpy as np

import matplotlib.pyplot as plt

logger = logging.getLogger("detectron2")
logging.basicConfig(level=logging.INFO)

import imgaug.augmenters as iaa
from imgaug.augmenters import (Sometimes, GaussianBlur,Add,AdditiveGaussianNoise, Multiply,CoarseDropout,Invert,pillike)

seq = iaa.Sequential([
    Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
    Sometimes(0.4, GaussianBlur((0., 3.))),
    Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),
    Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),
    Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),
    Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),
    Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
    Sometimes(0.3, Invert(0.2, per_channel=True)),
    Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
    Sometimes(0.5, Multiply((0.6, 1.4))),
    Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),
    Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
    ], random_order=True)

def color_aug(image_original):
    image = np.asarray(image_original, dtype=np.uint8).copy()
    image = image.transpose(1, 2, 0)
    image = seq(image=image)
    image = image.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image)

    original_image = image_original.permute(1, 2, 0).numpy()
    augmented_image = image_tensor.permute(1, 2, 0).numpy()

    original_image = original_image[:, :, ::-1]
    augmented_image = augmented_image[:, :, ::-1]

    return image_tensor
        
def do_train(cfg, model, aug, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[]))
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            if aug:
                for d in data:
                    d['image'] = color_aug(d['image'])

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def main(args):
    cfg = get_cfg()

    if args.model_type == "frcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    elif args.model_type == "retinanet":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))

    with open(args.dataset_config, 'r') as json_file:
        dataset_info = json.load(json_file)
        
    if dataset_info["dataset"] not in ["itodd", "itodd_random_texture"]:
        register_coco_instances(dataset_info["train_dataset_name"], {}, dataset_info["train_annotations"], dataset_info["train_images_dir"])
        register_coco_instances(dataset_info["test_dataset_name"], {}, dataset_info["test_annotations"], dataset_info["test_images_dir"])
    else:
        if dataset_info["dataset"] == "itodd":
            from configs.itodd_pbr import register_with_name_cfg
            register_with_name_cfg("itodd_pbr_train")   
        if dataset_info["dataset"] == "itodd_random_texture":
            from configs.itodd_random_texture_pbr import register_with_name_cfg
            register_with_name_cfg("itodd_random_texture_pbr_train")                         
        from configs.itodd_bop_test import register_with_name_cfg
        register_with_name_cfg("itodd_bop_test")

    output_dir = "/hdd/detectron2_models_new/" + args.model_type + "_" + dataset_info["dataset"] + "_model"

    if args.aug:
         output_dir += "_with_aug"
    
    print()
    print(output_dir)
    print()

    os.makedirs(output_dir, exist_ok=True)

    cfg.DATASETS.TRAIN = (dataset_info["train_dataset_name"])
    cfg.DATASETS.TEST = (dataset_info["test_dataset_name"])

    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = dataset_info["batch_size"]
    cfg.SOLVER.BASE_LR = 0.00025

    epochs = dataset_info["epochs"]

    single_iteration = 1 * cfg.SOLVER.IMS_PER_BATCH
    iterations_for_one_epoch = iterations_for_one_epoch = 50000 / single_iteration
    cfg.SOLVER.MAX_ITER = int(iterations_for_one_epoch * epochs)

    print(cfg.MODEL)
    if args.model_type == "frcnn":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = dataset_info["num_classes"]
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    else:
        cfg.MODEL.RETINANET.NUM_CLASSES = dataset_info["num_classes"]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01

    # Set the checkpoint saving options
    cfg.OUTPUT_DIR = output_dir  # Directory to save the checkpoints
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000  # Save a checkpoint every 100 iterations
    
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if args.mode == "eval":
        model_path = output_dir + "/model_final.pth"
        cfg.MODEL.WEIGHTS = model_path

        predictor = DefaultPredictor(cfg)

        if dataset_info["dataset"] not in ["itodd", "itodd_random_texture"]: 
            evaluator = COCOEvaluator(dataset_info["test_dataset_name"], cfg, False, output_dir=output_dir)
        else:
            evaluator = BOPEvaluator(dataset_info["test_dataset_name"], cfg, False, output_dir=output_dir)            

        val_loader = build_detection_test_loader(cfg, dataset_info["test_dataset_name"], mapper=DatasetMapper(cfg, is_train=False, augmentations=[]))
        metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
        print(metrics)
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Pretrained weights
        do_train(cfg, model, args.aug, resume=False)

if __name__ == "__main__":

    # e.g.: python frcnn_retnet_base.py --mode eval --model_type frcnn --dataset_config dataset_configs/lmo.json --aug False

    parser = argparse.ArgumentParser(description="Train/Evaluate a Detectron2 model")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Mode: 'train' or 'eval'")
    parser.add_argument("--model_type", type=str, default="frcnn", help="Type of model to use")
    parser.add_argument("--dataset_config", type=str, default="dataset_configs/lmo.json", help="Type of model to use")
    parser.add_argument("--aug", action="store_true", help="augmentations yes or no")

    args = parser.parse_args()
    main(args)