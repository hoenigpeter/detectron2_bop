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
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import json

class BOPEvaluator(COCOEvaluator):
    # ...
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = self.instances_to_coco_json(instances, input["image_id"], input["scene_im_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)
                
    def instances_to_coco_json(self, instances, img_id, scene_im_id):
        """
        Dump an "Instances" object to a modified COCO-format json that's used for evaluation.

        Args:
            instances (Instances):
            img_id (int): the image id

        Returns:
            list[dict]: list of modified json annotations in COCO format.
        """
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        results = []
        for k in range(num_instance):
            scene_id = int(scene_im_id.split("/")[0])

            result = {
                "scene_id": int(scene_id),
                "image_id": img_id,
                "category_id": classes[k] + 1,
                "bbox": boxes[k],
                "score": scores[k],
                "time": 0.2,
            }
            results.append(result)
        return results