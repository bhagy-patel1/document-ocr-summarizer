# app.py — Document Layout + OCR Pipeline
import os, json, time, cv2, argparse
import numpy as np
from pathlib import Path

from utils.layout import load_model, detect_layout, CLASSES
from utils.ocr import extract_text, ocr_region
from utils.table import extract_table
from utils.visualize import draw_results

CONFIG_PATH  = "config/config2.yaml"
WEIGHTS_PATH = "weights/model_final2.pth"

# BGR colors per class: text / title / list / table / figure
CLASS_COLORS = {
    "text"  : (0, 200, 0),
    "title" : (0, 0, 220),
    "list"  : (220, 220, 0),
    "table" : (0, 180, 220),
    "figure": (220, 100, 0),
}


def process_image(image_path: str, predictor, cfg,
                  ocr_processor, ocr_model,
                  save_annotated: bool = True) -> dict:
    """Run full pipeline on a single image. Returns structured dict."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    t0 = time.time()
    boxes, scores, classes = detect_layout(predictor, img)
    det_time = time.time() - t0

    result = {
        "file"       : image_path,
        "image_size" : {"width": img.shape[1], "height": img.shape[0]},
        "n_regions"  : len(boxes),
        "det_time_s" : round(det_time, 3),
        "regions"    : [],
        "titles"     : [],
        "text_blocks": [],
        "lists"      : [],
        "tables"     : [],
        "figures"    : [],
    }

    figure_count = 0
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = map(int, box)
        label = CLASSES[int(cls_id)]
        crop  = img[y1:y2, x1:x2]

        if label == "figure":
            # Save figure crop instead of running OCR
            fig_path = str(Path(image_path).stem) + f"_figure_{figure_count}.jpg"
            cv2.imwrite(str(Path("output") / fig_path), crop)
            text = f"[FIGURE saved: {fig_path}]"
            figure_count += 1
        elif label == "table":
            table_data = extract_table(crop) if ocr_processor is not False else {"rows": 0, "cols": 0, "cells": [], "raw_text": ""}
            text = table_data["raw_text"]
        elif ocr_processor is False:  # --no-ocr flag
            text = ""
        else:
            text = extract_text(crop)

        region = {
            "id"   : i,
            "label": label,
            "bbox" : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "score": float(round(score, 3)),
            "text" : text,
        }
        result["regions"].append(region)

        if label == "title":
            result["titles"].append(text)
        elif label == "text":
            result["text_blocks"].append(text)
        elif label == "list":
            result["lists"].append(text)
        elif label == "table":
            result["tables"].append({
                "bbox"    : region["bbox"],
                "rows"    : table_data.get("rows", 0) if label == "table" else 0,
                "cols"    : table_data.get("cols", 0) if label == "table" else 0,
                "cells"   : table_data.get("cells", []) if label == "table" else [],
                "raw_text": text,
            })
        elif label == "figure":
            result["figures"].append({"bbox": region["bbox"], "caption": ""})

    result["ocr_time_s"] = round(time.time() - t0 - det_time, 3)

    if save_annotated:
        annotated = draw_results(img.copy(), result["regions"], boxes, classes)
        out_path  = Path("output") / (Path(image_path).stem + "_annotated.jpg")
        cv2.imwrite(str(out_path), annotated)
        result["annotated_image"] = str(out_path)
        print(f"  Saved annotated: {out_path}")

    return result


def process_pdf(pdf_path: str, predictor, cfg,
                ocr_processor, ocr_model, dpi: int = 150) -> list:
    """Convert PDF pages to images and run pipeline on each."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Install: pip install pdf2image")
        return []

    print(f"Converting PDF: {pdf_path}  (dpi={dpi})")
    pages = convert_from_path(pdf_path, dpi=dpi)
    all_results = []

    for i, page_img in enumerate(pages):
        tmp = f"_tmp_page_{i:04d}.jpg"
        page_img.save(tmp, "JPEG", quality=95)
        print(f"\n--- Page {i+1}/{len(pages)} ---")
        try:
            res = process_image(tmp, predictor, cfg, ocr_processor, ocr_model)
            res["page"] = i + 1
            all_results.append(res)
        except Exception as e:
            print(f"  Error on page {i+1}: {e}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    return all_results


def print_result(result: dict):
    print("\n" + "=" * 60)
    print(f"  File    : {result['file']}")
    print(f"  Size    : {result['image_size']['width']} x {result['image_size']['height']}")
    print(f"  Regions : {result['n_regions']}")
    print(f"  Det time: {result['det_time_s']}s  |  OCR time: {result['ocr_time_s']}s")
    print("=" * 60)
    if result["titles"]:
        print(f"\n  TITLES ({len(result['titles'])}):")
        for t in result["titles"]:
            print(f"    >> {t[:100]}")
    if result["text_blocks"]:
        print(f"\n  TEXT BLOCKS ({len(result['text_blocks'])}):")
        for i, t in enumerate(result["text_blocks"][:3]):
            print(f"    [{i+1}] {t[:120]}...")
    if result["tables"]:
        print(f"\n  TABLES ({len(result['tables'])}):")
        for tb in result["tables"]:
            b = tb["bbox"]
            print(f"    bbox=({b['x1']},{b['y1']})-({b['x2']},{b['y2']})")
    if result["figures"]:
        print(f"\n  FIGURES ({len(result['figures'])}): detected")
    if result["lists"]:
        print(f"\n  LISTS ({len(result['lists'])}):")
        for l in result["lists"]:
            print(f"    - {l[:100]}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Layout + OCR Pipeline")
    parser.add_argument("input", help="Path to image (.jpg/.png) or PDF")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR (layout only)")
    parser.add_argument("--dpi",    type=int, default=150, help="DPI for PDF rendering")
    args = parser.parse_args()

    Path("output").mkdir(exist_ok=True)

    predictor, cfg = load_model(CONFIG_PATH, WEIGHTS_PATH)
    # PaddleOCR initializes lazily — no explicit load needed
    ocr_processor = False if args.no_ocr else None
    if args.no_ocr:
        print("[OCR] Skipped (--no-ocr)")

    if args.input.lower().endswith(".pdf"):
        results = process_pdf(args.input, predictor, cfg, None, None, dpi=args.dpi)
        out_json = Path("output") / (Path(args.input).stem + "_output.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(results)} pages → {out_json}")
        for r in results:
            print_result(r)
    else:
        result = process_image(args.input, predictor, cfg, ocr_processor, None)
        out_json = Path("output") / (Path(args.input).stem + "_output.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved → {out_json}")
        print_result(result)
