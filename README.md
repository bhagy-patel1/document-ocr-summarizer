# Document OCR Summarizer

A document layout analysis and OCR pipeline built with Detectron2 (Mask R-CNN) and PaddleOCR. Detects and extracts structured content — titles, text blocks, lists, tables, and figures — from document images and PDFs.

## Features

- Layout detection using a trained Mask R-CNN model (Detectron2)
- OCR via PaddleOCR optimized for printed document text
- Structured table extraction using line detection + projection splitting
- PDF support (page-by-page processing)
- Batch processing for entire folders
- Annotated output images + structured JSON per file

## Project Structure

```
document-ocr-pipeline/
├── app.py                  # Single image / PDF entry point
├── batch_process.py        # Batch folder processing
├── requirements.txt
├── config/
│   └── config.yaml         # Detectron2 model config
├── weights/
│   └── model_final.pth     # Trained Mask R-CNN weights (download separately)
└── utils/
    ├── layout.py           # Layout detection (Detectron2)
    ├── ocr.py              # OCR (PaddleOCR)
    ├── table.py            # Structured table extraction
    ├── sorting.py          # Box sorting utilities
    └── visualize.py        # Annotation drawing
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/bhagy-patel1/document-ocr-summarizer.git
cd document-ocr-summarizer
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Detectron2

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

### 5. Download model weights

Download `model_final.pth` and place it in the `weights/` folder.  
Download `config.yaml` and place it in the `config/` folder.

> Model trained on the [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) dataset.

### 6. (Optional) PDF support — install Poppler

- **Windows**: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) and add `bin/` to PATH
- **Linux**: `sudo apt install poppler-utils`
- **Mac**: `brew install poppler`

## Usage

### Single image

```bash
python app.py input/sample.jpg
```

### Single PDF

```bash
python app.py input/document.pdf --dpi 200
```

### Layout detection only (skip OCR)

```bash
python app.py input/sample.jpg --no-ocr
```

### Batch processing

```bash
python batch_process.py --input_dir ./input --output_dir ./output
```

## Output

Each run produces:
- `output/<name>_output.json` — structured JSON with regions, titles, text blocks, tables (with cell structure), figures
- `output/<name>_annotated.jpg` — image with colored bounding boxes per class

### JSON structure

```json
{
  "file": "input/sample.jpg",
  "n_regions": 12,
  "titles": ["6. Limitations"],
  "text_blocks": ["..."],
  "tables": [
    {
      "bbox": {"x1": 49, "y1": 99, "x2": 551, "y2": 280},
      "rows": 9,
      "cols": 10,
      "cells": [["Header1", "Header2", "..."], ["val", "val", "..."]],
      "raw_text": "Header1 | Header2 | ..."
    }
  ],
  "figures": [],
  "lists": []
}
```

## Classes

| Label  | Color  | Description        |
|--------|--------|--------------------|
| text   | Green  | Body text blocks   |
| title  | Red    | Section headings   |
| list   | Cyan   | Bullet/numbered lists |
| table  | Yellow | Data tables        |
| figure | Blue   | Images / diagrams  |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Detectron2
- PaddleOCR 2.7+
- OpenCV, Pillow, NumPy
