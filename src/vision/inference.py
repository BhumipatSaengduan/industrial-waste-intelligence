import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

WASTE_CLASSES = ['__background__', 'Metal', 'Mixed Waste', 'Paper-Cardboard', 'Plastic', 'Wood']
WASTE_CLASSES_NO_BG = ['Metal', 'Mixed Waste', 'Paper-Cardboard', 'Plastic', 'Wood']

# Color per class for visualization (RGB)
CLASS_COLORS = {
    'Metal'          : (176, 176, 176),
    'Mixed Waste'    : (233, 85,  0),
    'Paper-Cardboard': (247, 220, 111),
    'Plastic'        : (52,  152, 219),
    'Wood'           : (142, 90,  43)
}

def load_predictor(model_path, score_thresh=0.3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Build model with same architecture used in training
    num_classes = 6  # background + 5 waste classes

    model = maskrcnn_resnet50_fpn(weights=None)

    # Replace box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both raw state_dict and wrapped checkpoint
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Set score threshold
    model.roi_heads.score_thresh = score_thresh

    print(f'Model loaded on {device}')
    return model, device

def run_inference(model_tuple, image_path):
    model, device = model_tuple

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb   = img_bgr[:, :, ::-1].copy()
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    # Extract predictions
    masks   = outputs['masks'].squeeze(1).cpu().numpy()   # (N, H, W)
    labels  = outputs['labels'].cpu().numpy()              # (N,)
    scores  = outputs['scores'].cpu().numpy()              # (N,)
    boxes   = outputs['boxes'].cpu().numpy()               # (N, 4)

    # Binary masks at threshold 0.5
    masks_bin = (masks > 0.5).astype(np.uint8)

    instances = {
        'masks' : masks_bin,
        'labels': labels,
        'scores': scores,
        'boxes' : boxes,
        'count' : len(labels)
    }

    # Calculate area per class
    class_areas = {cls: 0 for cls in WASTE_CLASSES_NO_BG}

    for mask, label in zip(masks_bin, labels):
        if 1 <= label <= 5:
            cls_name = WASTE_CLASSES[label]
            class_areas[cls_name] += mask.sum()

    # Composition as % of total detected waste area
    total_waste_px = sum(class_areas.values())
    if total_waste_px > 0:
        composition = {
            cls: round(float(area) / total_waste_px * 100, 2)
            for cls, area in class_areas.items()
        }
    else:
        composition = {cls: 0.0 for cls in WASTE_CLASSES_NO_BG}

    # Draw annotated image
    annotated_img = draw_masks(img_rgb.copy(), masks_bin, labels, scores)

    return instances, composition, annotated_img

def draw_masks(img_rgb, masks, labels, scores, alpha=0.5):
    """
    Draw segmentation masks and labels on image.
    Args:
        img_rgb: numpy array (H, W, 3) RGB
        masks: binary masks (N, H, W)
        labels: class indices (N,)
        scores: confidence scores (N,)
        alpha: mask transparency
    Returns:
        annotated: numpy array (H, W, 3) RGB
    """
    annotated = img_rgb.copy()

    for mask, label, score in zip(masks, labels, scores):
        if 1 <= label <= 5:
            cls_name = WASTE_CLASSES[label]
            color    = CLASS_COLORS.get(cls_name, (255, 255, 255))

            # Draw colored mask
            colored_mask = np.zeros_like(img_rgb)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]

            annotated = np.where(
                mask[:, :, np.newaxis] > 0,
                (annotated * (1 - alpha) + colored_mask * alpha).astype(np.uint8),
                annotated
            )

            # Draw bounding box label
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                x1, y1 = int(xs.min()), int(ys.min())
                label_text = f'{cls_name} {score:.2f}'
                cv2.putText(
                    annotated, label_text, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )

    return annotated