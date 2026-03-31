# Model Card - Industrial Waste Mask R-CNN

## Model Overview
- **Architecture**: Mask R-CNN with ResNet-50 FPN backbone
- **Framework**: Detectron2 / PyTorch
- **Task**: Instance segmentation of industrial waste classes
- **Dataset**: Kinsei Sangyo Co., Ltd - incineration facility images (Japan)

## Training Details
| Parameter | Baseline | Fine-tuned (selected) |
|---|---|---|
| Pretrained weights | COCO | COCO → Baseline checkpoint |
| Iterations | 3,000 | 6,000 |
| Batch size | 4 | 4 |
| Base LR | 0.0005 | 0.0001 |
| ROI batch per image | 128 | 256 |
| Score threshold | 0.5 | 0.3 |
| Augmentation | Flip, Brightness, Contrast | + Rotation, Vertical flip |

## Dataset
| Split | Images | Annotations |
|---|---|---|
| Train | 1,008 | 17,988 (filtered) |
| Valid | 96 | ~1,373 |
| Test | 49 | - |
| Total | 1,153 | - |

Classes: Metal, Mixed Waste, Paper-Cardboard, Plastic, Wood
Note: Class wastes excluded - parent/default class with no annotations
Date range: 2024-08-07 to 2024-10-01 (55 days)

## Performance Metrics (Validation Set)
| Metric | Baseline | Fine-tuned |
|---|---|---|
| mAP@0.5 | 47.97% | 48.95% |
| mAP@0.5:0.95 | 32.62% | 32.98% |
| mAP@0.75 | 36.66% | 37.53% |

## Per-class AP (Fine-tuned)
| Class | AP | Status |
|---|---|---|
| Metal | 55.46% | ok |
| Mixed Waste | 30.95% | low |
| Paper-Cardboard | 37.77% | low |
| Plastic | 40.69% | low |
| Wood | 0.00% | critical |

## Known Limitations
1. Wood AP = 0.00 - high size variance and limited training samples
2. Overall mAP below target - achieved 48.95% vs target 60%
3. Single facility dataset - generalization not guaranteed
4. Dark images - 23% of training images have brightness < 50

## Inference Output Format
{
  "filename": "image.jpg",
  "instances": 54,
  "composition": {
    "Metal": 0.0,
    "Mixed Waste": 0.0,
    "Paper-Cardboard": 0.0,
    "Plastic": 56.02,
    "Wood": 43.98
  },
  "total_area_detected": 100.0
}

## Future Improvements
- Collect 300-500 additional Wood images
- Upgrade backbone to ResNet-101
- Train for 10,000+ iterations
- Implement real-time RTSP camera integration
- Edge deployment on NVIDIA Jetson

## Usage
    from src.vision.inference import load_predictor, run_inference
    predictor = load_predictor('models/final/waste_mask_rcnn_final.pth')
    instances, composition, annotated_img = run_inference(predictor, 'image.jpg')
    print(composition)

## Files
- models/final/waste_mask_rcnn_final.pth — selected model weights
- src/vision/inference.py — inference script
- notebooks/02_modeling.ipynb — full training notebook
