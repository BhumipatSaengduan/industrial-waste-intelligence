# Model Card — Industrial Waste Mask R-CNN (torchvision)

## Model Overview
- Architecture: Mask R-CNN with ResNet-50 FPN backbone
- Framework: torchvision / PyTorch (native Windows support)
- Task: Instance segmentation of industrial waste classes
- Dataset: Kinsei Sangyo Co., Ltd — incineration facility images (Japan)

## Training Details
| Parameter | Baseline | Fine-tuned (selected) |
|---|---|---|
| Pretrained weights | COCO (torchvision DEFAULT) | Baseline checkpoint |
| Epochs | 10 | 5 |
| Batch size | 2 | 2 |
| Base LR | 0.005 | 0.001 |
| Optimizer | SGD | SGD |
| Scheduler | StepLR (step=3, gamma=0.1) | StepLR (step=3, gamma=0.1) |
| Score threshold | 0.3 | 0.3 |

## Performance Metrics (Validation Set)
| Metric | Baseline | Fine-tuned |
|---|---|---|
| mAP@0.5 | 63.42% | 63.45% |

## Classes
Metal, Mixed Waste, Paper-Cardboard, Plastic, Wood
Note: composition sums to 100% of detected waste area

## Inference Output Format
    {
      "filename": "image.jpg",
      "instances": 14,
      "composition": {
        "Metal": 0.0,
        "Mixed Waste": 0.0,
        "Paper-Cardboard": 0.0,
        "Plastic": 56.02,
        "Wood": 43.98
      },
      "total_area_detected": 100.0
    }

## Usage
    from src.vision.inference import load_predictor, run_inference
    model_tuple = load_predictor('models/final/waste_maskrcnn_torchvision.pth')
    instances, composition, annotated_img = run_inference(model_tuple, 'image.jpg')

## Files
- models/final/waste_maskrcnn_torchvision.pth — selected model weights
- src/vision/inference.py — inference script (torchvision)
- notebooks/02_modeling.ipynb — full training notebook
