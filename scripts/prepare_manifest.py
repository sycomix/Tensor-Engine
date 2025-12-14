#!/usr/bin/env python3
"""Prepare a TSV manifest for Tensor Engine multimodal training.

This script converts common image-caption dataset formats into the manifest format
expected by `tensor_engine.ImageTextDataLoader`:

    /absolute/path/to/image.jpg\tCaption text

Supported inputs:

- COCO-style JSON ("images" + "annotations")
- JSONL (one JSON object per line) with configurable keys
- CSV/TSV with configurable column names

The output manifest is always written with absolute image paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Record:
    image_path: Path
    caption: str


def _sanitize_caption(text: str) -> str:
    # Keep manifest one-record-per-line and TSV-safe.
    return " ".join(text.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def _resolve_image_path(raw_path: str, images_root: Optional[Path]) -> Path:
    p = Path(raw_path)
    if not p.is_absolute():
        if images_root is None:
            raise ValueError(
                f"Image path '{raw_path}' is not absolute and --images-root was not provided."
            )
        p = images_root / p
    return p.expanduser().resolve()


def _iter_jsonl(path: Path, image_key: str, caption_key: str) -> Iterator[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")
            if image_key not in obj:
                raise ValueError(f"JSONL line {line_no} missing key '{image_key}'")
            if caption_key not in obj:
                raise ValueError(f"JSONL line {line_no} missing key '{caption_key}'")
            yield str(obj[image_key]), str(obj[caption_key])


def _iter_csv(path: Path, image_col: str, caption_col: str, delimiter: str) -> Iterator[Tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("CSV/TSV has no header row")
        if image_col not in reader.fieldnames:
            raise ValueError(f"CSV/TSV missing column '{image_col}', found: {reader.fieldnames}")
        if caption_col not in reader.fieldnames:
            raise ValueError(f"CSV/TSV missing column '{caption_col}', found: {reader.fieldnames}")
        for row_no, row in enumerate(reader, start=2):
            img = row.get(image_col)
            cap = row.get(caption_col)
            if img is None or cap is None:
                raise ValueError(f"Row {row_no} missing required columns")
            img_s = str(img).strip()
            cap_s = str(cap).strip()
            if not img_s or not cap_s:
                continue
            yield img_s, cap_s


def _iter_coco_json(path: Path) -> Iterator[Tuple[str, str]]:
    data: Any
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("COCO JSON root must be an object")
    images = data.get("images")
    annotations = data.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO JSON must contain 'images' (list) and 'annotations' (list)")

    id_to_file: Dict[int, str] = {}
    for img in images:
        if not isinstance(img, dict):
            continue
        img_id = img.get("id")
        file_name = img.get("file_name")
        if isinstance(img_id, int) and isinstance(file_name, str) and file_name:
            id_to_file[img_id] = file_name

    if not id_to_file:
        raise ValueError("COCO JSON contained no usable images entries")

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        caption = ann.get("caption")
        if not isinstance(image_id, int) or not isinstance(caption, str):
            continue
        file_name = id_to_file.get(image_id)
        if not file_name:
            continue
        yield file_name, caption


def _detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        return "jsonl"
    if ext in {".csv", ".tsv"}:
        return "csv" if ext == ".csv" else "tsv"
    if ext == ".json":
        # Could be COCO-style or generic JSON array/object; we only support COCO here.
        return "coco"
    raise ValueError(f"Unsupported metadata extension '{ext}' for file: {path}")


def _build_records(
        metadata_path: Path,
        images_root: Optional[Path],
        fmt: str,
        jsonl_image_key: str,
        jsonl_caption_key: str,
        csv_image_col: str,
        csv_caption_col: str,
) -> List[Record]:
    pairs: Iterable[Tuple[str, str]]
    if fmt == "jsonl":
        pairs = _iter_jsonl(metadata_path, jsonl_image_key, jsonl_caption_key)
    elif fmt == "csv":
        pairs = _iter_csv(metadata_path, csv_image_col, csv_caption_col, delimiter=",")
    elif fmt == "tsv":
        pairs = _iter_csv(metadata_path, csv_image_col, csv_caption_col, delimiter="\t")
    elif fmt == "coco":
        pairs = _iter_coco_json(metadata_path)
    else:
        raise ValueError(f"Unknown format '{fmt}'")

    out: List[Record] = []
    for raw_img, raw_cap in pairs:
        img_path = _resolve_image_path(raw_img, images_root)
        cap = _sanitize_caption(raw_cap)
        if not cap:
            continue
        out.append(Record(image_path=img_path, caption=cap))
    return out


def _write_manifest(records: List[Record], out_path: Path, verify_exists: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing: List[Path] = []
    if verify_exists:
        for r in records:
            if not r.image_path.exists():
                missing.append(r.image_path)
        if missing:
            sample = "\n".join(str(p) for p in missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} image files referenced by metadata do not exist. Sample:\n{sample}"
            )

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(f"{str(r.image_path)}\t{r.caption}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert image-caption datasets to Tensor Engine manifest.txt")
    parser.add_argument("--metadata", required=True, help="Path to COCO JSON, JSONL, CSV, or TSV metadata file")
    parser.add_argument(
        "--images-root",
        default=None,
        help="Root directory for relative image paths in metadata (required if metadata paths are relative)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output manifest path (TSV). Each line: /abs/path/to/image\tcaption",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap on number of rows written (0 = no cap)")

    # JSONL keys
    parser.add_argument("--jsonl-image-key", default="image_path", help="JSONL key for image path")
    parser.add_argument("--jsonl-caption-key", default="caption", help="JSONL key for caption")

    # CSV/TSV columns
    parser.add_argument("--csv-image-col", default="image_path", help="CSV/TSV column for image path")
    parser.add_argument("--csv-caption-col", default="caption", help="CSV/TSV column for caption")

    parser.add_argument("--no-verify-exists", action="store_true", help="Do not verify that image files exist")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    metadata_path = Path(args.metadata).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    images_root: Optional[Path]
    if args.images_root:
        images_root = Path(args.images_root).expanduser().resolve()
        if not images_root.exists():
            raise FileNotFoundError(f"--images-root does not exist: {images_root}")
    else:
        images_root = None

    out_path = Path(args.output).expanduser().resolve()

    fmt = _detect_format(metadata_path)
    _LOG.info("Detected metadata format: %s", fmt)

    records = _build_records(
        metadata_path=metadata_path,
        images_root=images_root,
        fmt=fmt,
        jsonl_image_key=args.jsonl_image_key,
        jsonl_caption_key=args.jsonl_caption_key,
        csv_image_col=args.csv_image_col,
        csv_caption_col=args.csv_caption_col,
    )

    if not records:
        raise ValueError("No usable records found in metadata")

    # Deterministic order for reproducibility.
    records.sort(key=lambda r: (str(r.image_path), r.caption))

    if args.max_rows and args.max_rows > 0:
        records = records[: args.max_rows]

    _LOG.info("Writing %d manifest rows to %s", len(records), out_path)
    _write_manifest(records, out_path, verify_exists=not args.no_verify_exists)

    _LOG.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
