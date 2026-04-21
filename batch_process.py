"""
batch_process.py
Run the Document OCR pipeline on an entire folder of images/PDFs.
Output: one JSON per file + a combined batch_summary.json.

Usage:
    python batch_process.py --input_dir ./input --output_dir ./output
"""
import os, json, time, argparse
from pathlib import Path

from app import process_image, process_pdf, print_result
from utils.layout import load_model
from utils.ocr import load_ocr_model

CONFIG_PATH  = "config/config.yaml"
WEIGHTS_PATH = "weights/model_final.pth"
SUPPORTED    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"}


def run_batch(input_dir: str, output_dir: str,
              skip_ocr: bool = False, dpi: int = 150):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(input_dir.iterdir())
             if f.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No supported files in {input_dir}")
        return

    print(f"Found {len(files)} files in {input_dir}")
    print("Loading models...")
    predictor, cfg = load_model(CONFIG_PATH, WEIGHTS_PATH)
    ocr_proc, ocr_mod = (None, None) if skip_ocr else load_ocr_model()

    summary  = []
    t_total  = time.time()

    for i, fpath in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]  {fpath.name}")
        t0 = time.time()
        try:
            if fpath.suffix.lower() == ".pdf":
                results = process_pdf(str(fpath), predictor, cfg,
                                      ocr_proc, ocr_mod, dpi=dpi)
                out_json = output_dir / (fpath.stem + "_output.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                for r in results:
                    print_result(r)
                entry = {"file": str(fpath), "pages": len(results),
                         "status": "ok", "time_s": round(time.time() - t0, 2)}
            else:
                result = process_image(str(fpath), predictor, cfg,
                                       ocr_proc, ocr_mod)
                # Move annotated image to output dir
                ann = Path(result.get("annotated_image", ""))
                if ann.exists() and ann.parent != output_dir:
                    ann.rename(output_dir / ann.name)
                out_json = output_dir / (fpath.stem + "_output.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print_result(result)
                entry = {
                    "file"   : str(fpath),
                    "regions": result["n_regions"],
                    "titles" : len(result["titles"]),
                    "text"   : len(result["text_blocks"]),
                    "tables" : len(result["tables"]),
                    "figures": len(result["figures"]),
                    "status" : "ok",
                    "time_s" : round(time.time() - t0, 2),
                }
        except Exception as e:
            print(f"  ERROR: {e}")
            entry = {"file": str(fpath), "status": f"error: {e}"}

        summary.append(entry)

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_files" : len(files),
            "total_time_s": round(time.time() - t_total, 2),
            "files"       : summary,
        }, f, indent=2)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Batch complete : {len(files)} files")
    print(f"  Total time     : {elapsed:.1f}s")
    print(f"  Summary        : {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="./input")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--no_ocr",  action="store_true")
    parser.add_argument("--dpi",     type=int, default=150)
    args = parser.parse_args()
    run_batch(args.input_dir, args.output_dir,
              skip_ocr=args.no_ocr, dpi=args.dpi)
