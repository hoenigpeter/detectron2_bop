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

import imgaug.augmenters as iaa
from imgaug.augmenters import (Sometimes, GaussianBlur,Add,AdditiveGaussianNoise, Multiply,CoarseDropout,Invert,pillike)

import numpy as np
import torch

def color_aug(image_original):
    image = np.asarray(image_original, dtype=np.uint8).copy()
    image = image.transpose(1, 2, 0)
    #image_np = image.cpu().numpy()
    image = seq(image=image)
    image = image.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image)

    original_image = image_original.permute(1, 2, 0).numpy()
    augmented_image = image_tensor.permute(1, 2, 0).numpy()

    original_image = original_image[:, :, ::-1]
    augmented_image = augmented_image[:, :, ::-1]


logger = logging.getLogger("detectron2")

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


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret

def do_train(args, cfg):
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)

def main(args):
    cfg = LazyConfig.load("projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_s_3x.py")
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    print(cfg)
    register_coco_instances("tless_pbr_train", {}, "datasets/BOP_DATASETS/tless/tless_annotations_train.json", "datasets/BOP_DATASETS/tless/train_pbr")
    register_coco_instances("tless_bop_test_primesense", {}, "datasets/BOP_DATASETS/tless/tless_annotations_test.json", "datasets/BOP_DATASETS/tless/test_primesense")

    output_dir = "./mvit2_tless_v2_output"
    os.makedirs(output_dir, exist_ok=True)

    cfg.dataloader.train.dataset.names = "tless_pbr_train"
    cfg.dataloader.test.dataset.names = "tless_bop_test_primesense"
    cfg.dataloader.train.total_batch_size = 4
    cfg.train.output_dir = output_dir
    cfg.model.roi_heads.num_classes = 30
    cfg.dataloader.train.mapper.augmentations = []
    cfg.dataloader.test.mapper.augmentations = []
    cfg.train.eval_period = 10000

    epochs = 30 

    single_iteration = 1 * cfg.dataloader.train.total_batch_size
    iterations_for_one_epoch = iterations_for_one_epoch = 50000 / single_iteration

    cfg.train.max_iter = int(iterations_for_one_epoch * epochs)  # Adjust according to your requirements   

    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


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
