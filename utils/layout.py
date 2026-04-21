# layout.py — Detectron2 layout detection utilities
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

CLASSES = ["text", "title", "list", "table", "figure"]
SCORE_THR = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config_path: str, weights_path: str):
    """Load Mask R-CNN layout model."""
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = DEVICE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THR
    # Register metadata so Visualizer has class names
    if "doc_layout" not in MetadataCatalog:
        MetadataCatalog.get("doc_layout").set(thing_classes=CLASSES)
    predictor = DefaultPredictor(cfg)
    print(f"[Layout] Mask R-CNN loaded ({DEVICE})")
    return predictor, cfg


def _expand_boxes(boxes: np.ndarray, image_shape: tuple, pad: int = 10) -> np.ndarray:
    """Expand each box by `pad` pixels on all sides, clamped to image bounds."""
    h, w = image_shape[:2]
    expanded = boxes.copy().astype(int)
    expanded[:, 0] = np.clip(expanded[:, 0] - pad, 0, w)  # x1
    expanded[:, 1] = np.clip(expanded[:, 1] - pad, 0, h)  # y1
    expanded[:, 2] = np.clip(expanded[:, 2] + pad, 0, w)  # x2
    expanded[:, 3] = np.clip(expanded[:, 3] + pad, 0, h)  # y2
    return expanded


def detect_layout(predictor, image: np.ndarray, expand_pad: int = 10):
    """
    Run layout detection on an image.
    Returns (boxes, scores, classes) as numpy arrays, sorted top→bottom.
    Boxes are expanded by expand_pad pixels to avoid edge-clipping in OCR.
    """
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy().astype(int)
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    if len(boxes) == 0:
        return boxes, scores, classes

    # Expand boxes to capture edge characters
    boxes = _expand_boxes(boxes, image.shape, pad=expand_pad)

    # Sort top→bottom (reading order)
    order = np.argsort(boxes[:, 1])
    return boxes[order], scores[order], classes[order]
