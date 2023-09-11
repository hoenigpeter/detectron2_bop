#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

logger = logging.getLogger("detectron2")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os

from evaluators.bop_evaluator import BOPEvaluator

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        print(cfg.dataloader)
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def main(args):
    cfg = LazyConfig.load("./projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_s_3x.py")
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    print(cfg)

    #LazyConfig.save(cfg, "tmp.yaml")
    #register_coco_instances("itodd_pbr_train", {}, "datasets/itodd/itodd_annotations_train.json", "datasets/itodd/train_pbr")
    #register_coco_instances("itodd_bop_test", {}, "datasets/itodd/itodd_annotations_test.json", "datasets/itodd/test_primesense")
    from configs.itodd_bop_test import register_with_name_cfg
    register_with_name_cfg("itodd_bop_test")
    print("dataset catalog: ", DatasetCatalog.list())

    output_dir = "./mvit2_itodd_random_texture_output"
    os.makedirs(output_dir, exist_ok=True)

    #cfg.dataloader.train.dataset.names = "itodd_pbr_train"
    cfg.dataloader.test.dataset.names = "itodd_bop_test"
    #cfg.dataloader.test.mapper.augmentations = []
    cfg.dataloader.evaluator = BOPEvaluator("itodd_bop_test", cfg, False, output_dir=output_dir)
    #cfg.dataloader.train.total_batch_size = 4

    cfg.model.roi_heads.num_classes = 28
    cfg.OUTPUT_DIR = output_dir

    #cfg.train.eval_period = 1000000

    epochs = 10 

    single_iteration = 1 * cfg.dataloader.train.total_batch_size
    iterations_for_one_epoch = iterations_for_one_epoch = 50000 / single_iteration

    cfg.train.max_iter = int(iterations_for_one_epoch * epochs)  # Adjust according to your requirements   

    cfg.train.output_dir = output_dir

    #default_setup(cfg, args)
    cfg.train.init_checkpoint = "./mvit2_itodd_random_texture_output/model_final.pth"

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    print(do_test(cfg, model))

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
