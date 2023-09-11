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

        print(img_id)
        print(scene_im_id)

        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        #has_mask = instances.has("pred_masks")
        has_mask = False
        if has_mask:
            # Use RLE to encode the masks, because they are too large and take memory
            # since this evaluator stores outputs of the entire dataset
            rles = [
                mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                for mask in instances.pred_masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

        has_keypoints = instances.has("pred_keypoints")
        if has_keypoints:
            keypoints = instances.pred_keypoints

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
            if has_mask:
                # Modify the mask format here (example: convert to grayscale)
                modified_mask = np.mean(instances.pred_masks[k].numpy(), axis=2, keepdims=True)
                rle = mask_utils.encode(np.array(modified_mask, order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                result["segmentation"] = rle
            if has_keypoints:
                # In COCO annotations,
                # keypoints coordinates are pixel indices.
                # However, our predictions are floating-point coordinates.
                # Therefore, we subtract 0.5 to be consistent with the annotation format.
                # This is the inverse of data loading logic in `datasets/coco.py`.
                keypoints[k][:, :2] -= 0.5
                result["keypoints"] = keypoints[k].flatten().tolist()
            results.append(result)
        return results