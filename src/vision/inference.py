import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

WASTE_CLASSES = ['Metal', 'Mixed Waste', 'Paper-Cardboard', 'Plastic', 'Wood']

def load_predictor(model_path, score_thresh=0.3):
    """Load fine-tuned Mask R-CNN predictor from checkpoint."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    cfg.MODEL.WEIGHTS = model_path
    return DefaultPredictor(cfg)

def run_inference(predictor, image_path):
    """
    Run inference on a single image.
    Args:
        predictor: Detectron2 DefaultPredictor
        image_path: path to input image
    Returns:
        instances: raw Detectron2 instances
        composition: dict of {class_name: area_percentage of detected waste}
        annotated_img: numpy array with masks drawn
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = img[:, :, ::-1]

    outputs   = predictor(img)
    instances = outputs["instances"].to("cpu")

    class_areas = {cls: 0 for cls in WASTE_CLASSES}

    if len(instances) > 0:
        masks   = instances.pred_masks.numpy()
        classes = instances.pred_classes.numpy()
        for mask, cls_id in zip(masks, classes):
            if cls_id < len(WASTE_CLASSES):
                class_areas[WASTE_CLASSES[cls_id]] += mask.sum()

    # Calculate proportion relative to total detected waste area
    total_waste_px = sum(class_areas.values())
    if total_waste_px > 0:
        composition = {
            cls: round(float(area) / total_waste_px * 100, 2)
            for cls, area in class_areas.items()
        }
    else:
        composition = {cls: 0.0 for cls in WASTE_CLASSES}

    metadata  = MetadataCatalog.get("waste_train")
    v         = Visualizer(img_rgb, metadata=metadata,
                           scale=1.0, instance_mode=ColorMode.IMAGE_BW)
    annotated = v.draw_instance_predictions(instances)

    return instances, composition, annotated.get_image()
