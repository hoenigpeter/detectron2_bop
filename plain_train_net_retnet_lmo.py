import logging
import os
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import default_writers
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data import DatasetCatalog
from detectron2.model_zoo import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader

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
        
def do_train(cfg, model, resume=False):
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


if __name__ == "__main__":
    from configs.lm_pbr import register_with_name_cfg
    register_with_name_cfg("lmo_pbr_train")
    from configs.lmo_bop_test import register_with_name_cfg
    register_with_name_cfg("lmo_bop_test")

    print("dataset catalog: ", DatasetCatalog.list())

    # Create a Detectron2 config
    # Add a directory to save the model checkpoints
    output_dir = "./retnet_lmo_model_with_aug"
    os.makedirs(output_dir, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    
    cfg.DATASETS.TRAIN = ("lmo_pbr_train",)
    cfg.DATASETS.TEST = ("lmo_bop_test",)
    cfg.DATALOADER.NUM_WORKERS = 4  # Adjust according to your system setup
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Pretrained weights
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025

    epochs = 30

    single_iteration = 1 * cfg.SOLVER.IMS_PER_BATCH
    iterations_for_one_epoch = 50000 / single_iteration

    cfg.SOLVER.MAX_ITER = int(iterations_for_one_epoch * epochs)  # Adjust according to your requirements
    cfg.MODEL.RETINANET.NUM_CLASSES = 8  # Adjust according to your dataset


    # Set the checkpoint saving options
    cfg.OUTPUT_DIR = output_dir  # Directory to save the checkpoints
    cfg.SOLVER.CHECKPOINT_PERIOD = 50000  # Save a checkpoint every 100 iterations
    
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    do_train(cfg, model, resume=False)