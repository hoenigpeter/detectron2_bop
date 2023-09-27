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
    #image_np = image.cpu().numpy()
    image = seq(image=image)
    image = image.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image)

    original_image = image_original.permute(1, 2, 0).numpy()
    augmented_image = image_tensor.permute(1, 2, 0).numpy()

    original_image = original_image[:, :, ::-1]
    augmented_image = augmented_image[:, :, ::-1]

    return image_tensor

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
        for d in data:
            d['image'] = color_aug(d['image'])        
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
    #trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    trainer = SimpleTrainer(model, train_loader, optim)
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
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    register_coco_instances("tless_random_texture_pbr_train", {}, "datasets/tless_random_texture/tless_random_texture_annotations_train.json", "datasets/tless_random_texture/train_pbr")
    register_coco_instances("tless_bop_test_primesense", {}, "datasets/tless/tless_annotations_test.json", "datasets/tless/test_primesense")

    output_dir = "./mvit2_tless_random_texture_with_aug"
    os.makedirs(output_dir, exist_ok=True)

    cfg.dataloader.train.dataset.names = "tless_random_texture_pbr_train"
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
