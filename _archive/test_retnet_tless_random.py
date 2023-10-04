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
from detectron2.utils.visualizer import Visualizer
import cv2

if __name__ == "__main__":
    from configs.tless_bop_test import register_with_name_cfg
    register_with_name_cfg("tless_bop_test_primesense")

    print("dataset catalog: ", DatasetCatalog.list())

    # Create a Detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))

    # Create a Detectron2 config
    # Add a directory to save the model checkpoints
    output_dir = "./retinanet_tless_random_texture_model_with_aug"
    model_path = output_dir + "/model_final.pth"
    cfg.MODEL.WEIGHTS = model_path
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # Set the desired threshold for detection
    cfg.MODEL.RETINANET.NUM_CLASSES = 30  # Adjust according to your dataset

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("tless_bop_test_primesense")
    dataset_dicts = DatasetCatalog.get("tless_bop_test_primesense")

    # Evaluate the model on the lm_bop_test dataset
    evaluator = COCOEvaluator("tless_bop_test_primesense", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "tless_bop_test_primesense")
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(metrics)

    # Save the images with detection results to the "output_images" folder
    # output_folder = "tless_output_images"
    # os.makedirs(output_folder, exist_ok=True)

    # for idx, d in enumerate(dataset_dicts):
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     output_file = os.path.join(output_folder, f"detection_{idx}.jpg")
    #     cv2.imwrite(output_file, v.get_image()[:, :, ::-1])