# ocr.py — PaddleOCR-based OCR utilities (optimized for document text)
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
from paddleocr import PaddleOCR

# Single shared instance — load once
_ocr = None

def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            det_db_thresh=0.3,
            det_db_box_thresh=0.4,
            det_db_unclip_ratio=1.6,   # expand detected boxes slightly
            rec_batch_num=8,            # batch for speed
            use_gpu=False,
            show_log=False,
        )
    return _ocr


def _preprocess(crop: np.ndarray) -> np.ndarray:
    """Upscale tiny crops, denoise, sharpen — improves PaddleOCR accuracy."""
    h, w = crop.shape[:2]
    # Upscale if too small
    if h < 48:
        scale = 48 / h
        crop = cv2.resize(crop, (int(w * scale), 48), interpolation=cv2.INTER_CUBIC)
    # Denoise
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def extract_text(crop: np.ndarray, *args, conf_thresh: float = 0.6) -> str:
    """
    Extract text from a BGR crop using PaddleOCR.
    Extra *args accepted so callers passing (processor, model) still work.
    """
    if crop is None or crop.size == 0:
        return ""

    processed = _preprocess(crop)
    ocr = _get_ocr()
    result = ocr.ocr(processed, cls=True)

    lines = []
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2:
                text, conf = line[1]
                if conf >= conf_thresh and text.strip():
                    lines.append(text.strip())

    return " ".join(lines)


# Alias so app.py import of ocr_region still works
def ocr_region(crop: np.ndarray, *args, **kwargs) -> str:
    return extract_text(crop, conf_thresh=kwargs.get("conf_thresh", 0.6))
