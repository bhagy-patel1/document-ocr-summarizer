<div align="center">

# 📄 Document OCR Summarizer

**Intelligent document layout analysis and OCR extraction pipeline**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Detectron2](https://img.shields.io/badge/Detectron2-Mask_R--CNN-0064C8?style=flat-square)](https://github.com/facebookresearch/detectron2)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7+-003CDB?style=flat-square)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

Detects and extracts structured content — titles, text blocks, lists, tables, and figures — from document images and PDFs using a Mask R-CNN model trained on PubLayNet.

[Features](#-features) · [Architecture](#-architecture) · [Setup](#-setup) · [Usage](#-usage) · [Output](#-output-format) · [Classes](#-layout-classes)

</div>

---

## ✨ Features

| Feature | Details |
|---|---|
| **Layout Detection** | Mask R-CNN (Detectron2) trained on PubLayNet — 5 region classes |
| **OCR Engine** | PaddleOCR 2.7 with denoising + sharpening preprocessing |
| **Table Extraction** | Line detection + projection splitting → structured cell grid |
| **PDF Support** | Page-by-page rendering via `pdf2image` + Poppler |
| **Batch Processing** | Folder-level pipeline with consolidated `batch_summary.json` |
| **Annotated Output** | Colored bounding boxes per class saved as `.jpg` |
| **Structured JSON** | Per-file JSON with regions, titles, tables (with cell structure), figures |

---

## 🏗 Architecture

```
Input (image / PDF)
        │
        ▼
┌───────────────────┐
│  Layout Detection  │  ← Detectron2 Mask R-CNN (PubLayNet weights)
│  detect_layout()  │     Outputs: boxes, scores, class IDs
└────────┬──────────┘
         │  sorted top → bottom (reading order)
         ▼
┌────────────────────────────────────────────┐
│              Region Router                  │
│                                            │
│  title / text / list → PaddleOCR          │
│  table               → extract_table()    │
│  figure              → save crop as .jpg  │
└────────────────────────────────────────────┘
         │
         ▼
┌───────────────────┐
│   Structured JSON  │  + Annotated image
└───────────────────┘
```

**Table extraction pipeline:**
```
Table crop → Morphological line detection (H + V)
           → Projection-based fallback (if no ruled lines)
           → PaddleOCR per cell → row-major cell grid
```

---

## 📁 Project Structure

```
document-ocr-summarizer/
│
├── app.py                  # Single image / PDF entry point
├── batch_process.py        # Batch folder processing
├── requirements.txt
│
├── config/
│   ├── config.yaml         # CPU inference config (Faster R-CNN)
│   └── config2.yaml        # GPU training config (Mask R-CNN, used in app.py)
│
├── weights/
│   └── model_final.pth     # Trained weights — download separately
│
├── utils/
│   ├── layout.py           # Detectron2 model loading + inference
│   ├── ocr.py              # PaddleOCR wrapper with preprocessing
│   ├── table.py            # Structured table extraction
│   ├── sorting.py          # Bounding box sort utilities
│   └── visualize.py        # Annotated image rendering
│
├── input/                  # Place input images / PDFs here
└── output/                 # JSON results + annotated images written here
```

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/bhagy-patel1/document-ocr-summarizer.git
cd document-ocr-summarizer
```

### 2. Virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Detectron2

Detectron2 must be installed separately (not on PyPI):

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

> **Windows users:** Pre-built wheels are available at [detectron2 releases](https://github.com/facebookresearch/detectron2/releases). Building from source requires Visual Studio Build Tools.

### 5. Download model weights

Download `model_final.pth` and place it in `weights/`:

```
weights/
└── model_final.pth    ← trained on PubLayNet dataset
```

> The model uses ResNet-50 + FPN backbone with 5 output classes: `text`, `title`, `list`, `table`, `figure`.

### 6. PDF support — install Poppler

Required only if processing PDF files:

| OS | Command |
|---|---|
| **Windows** | Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases), add `bin/` to `PATH` |
| **Linux** | `sudo apt install poppler-utils` |
| **macOS** | `brew install poppler` |

---

## 🚀 Usage

### Single image

```bash
python app.py input/sample.jpg
```

### Single PDF

```bash
python app.py input/document.pdf --dpi 200
```

> Higher DPI improves OCR accuracy at the cost of speed. Default is `150`.

### Layout detection only (skip OCR)

```bash
python app.py input/sample.jpg --no-ocr
```

### Batch processing

```bash
python batch_process.py --input_dir ./input --output_dir ./output
```

**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.pdf`

**Batch options:**

```bash
python batch_process.py \
  --input_dir  ./input   \
  --output_dir ./output  \
  --no_ocr               \   # skip OCR (layout only)
  --dpi 150                  # PDF render DPI
```

---

## 📦 Output Format

Each run writes two files per input:

```
output/
├── sample_output.json        # structured extraction result
└── sample_annotated.jpg      # image with colored region boxes
```

And for batch runs:

```
output/
└── batch_summary.json        # aggregate stats across all files
```

### JSON structure

```json
{
  "file": "input/sample.jpg",
  "image_size": { "width": 601, "height": 792 },
  "n_regions": 12,
  "det_time_s": 2.45,
  "ocr_time_s": 1.83,
  "titles": ["6. Limitations"],
  "text_blocks": ["Measuring rapid changes in renal function..."],
  "lists": [],
  "tables": [
    {
      "bbox": { "x1": 39, "y1": 91, "x2": 566, "y2": 289 },
      "rows": 9,
      "cols": 10,
      "cells": [
        ["Header1", "Header2", "..."],
        ["val",     "val",     "..."]
      ],
      "raw_text": "Header1 | Header2 | ..."
    }
  ],
  "figures": [
    { "bbox": { "x1": 88, "y1": 81, "x2": 511, "y2": 661 }, "caption": "" }
  ],
  "regions": [
    {
      "id": 0,
      "label": "title",
      "bbox": { "x1": 298, "y1": 357, "x2": 389, "y2": 390 },
      "score": 0.995,
      "text": "6. Limitations"
    }
  ],
  "annotated_image": "output/sample_annotated.jpg"
}
```

### batch_summary.json

```json
{
  "total_files": 22,
  "total_time_s": 87.17,
  "files": [
    {
      "file": "input/sample.jpg",
      "regions": 12, "titles": 1, "text": 10,
      "tables": 1,  "figures": 0,
      "status": "ok", "time_s": 4.27
    }
  ]
}
```

---

## 🎨 Layout Classes

| Label | Color | Class ID | Description |
|---|---|---|---|
| `text` | 🟩 Green | 0 | Body paragraphs |
| `title` | 🟥 Red | 1 | Section headings |
| `list` | 🟧 Orange | 2 | Bullet / numbered lists |
| `table` | 🟦 Teal | 3 | Data tables |
| `figure` | 🟪 Purple | 4 | Images, charts, diagrams |

---

## 🧩 Requirements

```
torch==2.0.1
torchvision==0.15.2
paddlepaddle==2.6.2
paddleocr==2.7.0.0
opencv-python==4.6.0.66
Pillow==10.0.1
numpy==1.26.4
pdf2image==1.16.3
fvcore==0.1.5.post20221221
pycocotools==2.0.11
```

> Detectron2 is installed separately — see [Setup](#-setup).

---

## 📊 Benchmark (sample batch — 22 files)

| Metric | Value |
|---|---|
| Total files processed | 22 |
| Total time | 87.2s |
| Avg time per image | ~3.96s |
| Avg regions per image | ~11 |
| Files with tables | 6 |
| Files with figures | 9 |

---

## 🗂 Dataset

Model trained on **[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)** — a large dataset of scientific document images with pixel-level layout annotations across 5 classes.

---

<div align="center">

Built with [Detectron2](https://github.com/facebookresearch/detectron2) · [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) · [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)

</div>
