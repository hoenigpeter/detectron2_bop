import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    TrainerBase,
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
from detectron2.data import DatasetMapper, build_detection_train_loader

logger = logging.getLogger("detectron2")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os

import imgaug.augmenters as iaa
from imgaug.augmenters import (Sometimes, GaussianBlur,Add,AdditiveGaussianNoise, Multiply,CoarseDropout,Invert,pillike)
import torch
import numpy as np
import time
from detectron2.utils.events import get_event_storage
from torch.nn.parallel import DataParallel, DistributedDataParallel

class CustomSimpleTrainer(TrainerBase):
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
     
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

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
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load("projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_s_3x.py")
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    register_coco_instances("lmo_random_texture_all_pbr_train", {}, "datasets/BOP_DATASETS/lmo_random_texture_all/lmo_random_texture_all_annotations_train.json", "datasets/BOP_DATASETS/lmo_random_texture_all/train_pbr")
    register_coco_instances("lmo_bop_test", {}, "datasets/BOP_DATASETS/lmo/lmo_annotations_test.json", "datasets/BOP_DATASETS/lmo/test")

    output_dir = "./mvit2_lmo_random_texture_output"
    os.makedirs(output_dir, exist_ok=True)

    cfg.dataloader.train.dataset.names = "lmo_random_texture_all_pbr_train"
    cfg.dataloader.test.dataset.names = "lmo_bop_test"
    cfg.dataloader.train.total_batch_size = 4
    cfg.train.output_dir = output_dir
    cfg.model.roi_heads.num_classes = 8

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
