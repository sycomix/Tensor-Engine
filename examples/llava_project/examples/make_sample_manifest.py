#!/usr/bin/env python3
"""Generate a small manifest pointing at the bundled sample image.

This is meant as a *smoke* path so the llava_project can be exercised end-to-end
(train on a real image file + generate with --image) without any external dataset.

It writes an absolute-path manifest line of the form:
    /abs/path/to/sample_image.ppm\t<caption>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a sample manifest for llava_project")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "data" / "sample_manifest.txt"),
        help="Output manifest path (TSV)",
    )
    parser.add_argument(
        "--caption",
        default="A tiny sample image with colored blocks.",
        help="Caption to write for the sample image",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    img_path = (data_dir / "sample_image.ppm").resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Bundled sample image not found: {img_path}")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    caption = " ".join(str(args.caption).replace("\t", " ").split())

    out_path.write_text(f"{str(img_path)}\t{caption}\n", encoding="utf-8")
    print(f"Wrote sample manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
