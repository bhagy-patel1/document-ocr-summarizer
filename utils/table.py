# table.py — structured table extraction using line detection + PaddleOCR
import cv2
import numpy as np
from utils.ocr import extract_text


def _detect_lines(gray: np.ndarray, axis: int, min_length_ratio: float = 0.2):
    """Detect horizontal (axis=0) or vertical (axis=1) lines via morphology."""
    size = gray.shape[1] if axis == 0 else gray.shape[0]
    kernel_len = max(10, int(size * min_length_ratio))

    if axis == 0:  # horizontal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    else:          # vertical
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    eroded   = cv2.erode(binary, kernel, iterations=1)
    dilated  = cv2.dilate(eroded, kernel, iterations=1)
    return dilated


def _line_positions(line_mask: np.ndarray, axis: int, gap: int = 5):
    """Return sorted pixel positions where lines exist."""
    projection = np.sum(line_mask, axis=axis)   # collapse along axis
    positions  = np.where(projection > line_mask.shape[axis] * 0.2)[0]
    if len(positions) == 0:
        return []
    # Cluster close positions into single line coords
    clusters, group = [], [positions[0]]
    for p in positions[1:]:
        if p - group[-1] <= gap:
            group.append(p)
        else:
            clusters.append(int(np.mean(group)))
            group = [p]
    clusters.append(int(np.mean(group)))
    return clusters


def _split_by_projection(gray: np.ndarray, axis: int, min_gap: int = 6):
    """
    Fallback: split by whitespace projection when no ruled lines found.
    axis=0 → find row splits (project along cols)
    axis=1 → find col splits (project along rows)
    """
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    if axis == 0:
        density = np.sum(binary, axis=1)   # per row
    else:
        density = np.sum(binary, axis=0)   # per col

    # Smooth density to avoid noise
    kernel = np.ones(3) / 3
    density = np.convolve(density.astype(float), kernel, mode='same')

    in_gap   = density < 8
    splits   = []
    gap_start = None
    for i, empty in enumerate(in_gap):
        if empty and gap_start is None:
            gap_start = i
        elif not empty and gap_start is not None:
            if i - gap_start >= min_gap:
                splits.append((gap_start + i) // 2)
            gap_start = None
    return splits


def extract_table(crop_bgr: np.ndarray) -> dict:
    """
    Extract a table crop into a structured dict:
    {
      "rows": int,
      "cols": int,
      "cells": [[str, ...], ...],   # row-major
      "raw_text": str               # flat fallback
    }
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {"rows": 0, "cols": 0, "cells": [], "raw_text": ""}

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Detect ruled lines ────────────────────────────────────────────────
    h_mask = _detect_lines(gray, axis=0)
    v_mask = _detect_lines(gray, axis=1)

    row_lines = _line_positions(h_mask, axis=1)   # horizontal → y coords
    col_lines = _line_positions(v_mask, axis=0)   # vertical   → x coords

    # ── Fallback to projection if no lines found ──────────────────────────
    if len(row_lines) < 2:
        row_lines = _split_by_projection(gray, axis=0)
        row_lines = [0] + row_lines + [h]
    else:
        if row_lines[0] > 5:
            row_lines = [0] + row_lines
        if row_lines[-1] < h - 5:
            row_lines = row_lines + [h]

    if len(col_lines) < 2:
        col_lines = _split_by_projection(gray, axis=1)
        col_lines = [0] + col_lines + [w]
    else:
        if col_lines[0] > 5:
            col_lines = [0] + col_lines
        if col_lines[-1] < w - 5:
            col_lines = col_lines + [w]

    # ── OCR each cell ─────────────────────────────────────────────────────
    cells = []
    for r in range(len(row_lines) - 1):
        row_cells = []
        y1, y2 = row_lines[r], row_lines[r + 1]
        if y2 - y1 < 4:          # skip hairline gaps
            continue
        for c in range(len(col_lines) - 1):
            x1, x2 = col_lines[c], col_lines[c + 1]
            if x2 - x1 < 4:
                continue
            cell_crop = crop_bgr[y1:y2, x1:x2]
            text = extract_text(cell_crop, conf_thresh=0.4)
            row_cells.append(text)
        if any(row_cells):        # skip fully empty rows
            cells.append(row_cells)

    raw_text = " | ".join(
        " ".join(cell for cell in row if cell)
        for row in cells
    )

    return {
        "rows"    : len(cells),
        "cols"    : max((len(r) for r in cells), default=0),
        "cells"   : cells,
        "raw_text": raw_text,
    }
