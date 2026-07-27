#!/usr/bin/env python3
"""Generate the canonical Tensor API conformance inventory from Rust sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TENSOR_SOURCE = ROOT / "src" / "tensor.rs"
OVERRIDES = ROOT / "conformance" / "tensor_api_overrides.json"
JSON_OUTPUT = ROOT / "conformance" / "tensor_api_matrix.json"
MARKDOWN_OUTPUT = ROOT / "conformance" / "TENSOR_API_MATRIX.md"

NON_OPERATION_APIS = {
    "apply", "backward", "broadcast_shapes", "build_topo", "detach", "dtype",
    "is_same", "lock", "new", "new_pooled", "new_with_dtype", "ones",
    "ones_pooled", "randn", "requires_grad", "set_requires_grad", "shape",
    "to_f32_array", "to_vec", "validate_quantized_weights_2d", "zero_grad",
    "zeros", "zeros_pooled", "from_scalar",
}
REVIEW_FIELDS = ("backward", "broadcasting", "dtype", "rank", "cpu", "wgpu", "cuda")


def extract_public_functions(text: str) -> list[dict[str, object]]:
    lines = text.splitlines()
    functions: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        match = re.match(r"\s*pub fn\s+([A-Za-z_][A-Za-z0-9_]*)", lines[index])
        if not match:
            index += 1
            continue
        name = match.group(1)
        signature_lines = [lines[index].strip()]
        while "{" not in signature_lines[-1] and index + 1 < len(lines):
            index += 1
            signature_lines.append(lines[index].strip())
        signature = " ".join(signature_lines).split("{", 1)[0].strip()
        functions.append(
            {
                "name": name,
                "line": index + 2 - len(signature_lines),
                "signature": re.sub(r"\s+", " ", signature),
                "kind": "helper" if name in NON_OPERATION_APIS else "operation",
            }
        )
        index += 1
    return functions


def test_references(name: str) -> list[str]:
    patterns = (
        re.compile(rf"\.{re.escape(name)}\s*\("),
        re.compile(rf"\bTensor::{re.escape(name)}\s*\("),
    )
    matches: list[str] = []
    for base in (ROOT / "tests", ROOT / "src"):
        for path in sorted(base.rglob("*.rs")):
            if path == TENSOR_SOURCE:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in patterns):
                matches.append(path.relative_to(ROOT).as_posix())
    return sorted(set(matches))


def build_matrix() -> dict[str, object]:
    source = TENSOR_SOURCE.read_text(encoding="utf-8")
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    functions = extract_public_functions(source)
    unknown_overrides = sorted(set(overrides) - {entry["name"] for entry in functions})
    if unknown_overrides:
        raise ValueError(f"overrides reference unknown APIs: {', '.join(unknown_overrides)}")

    entries = []
    for function in functions:
        name = str(function["name"])
        references = test_references(name)
        entry: dict[str, object] = {
            **function,
            "forward": "implemented",
            "backward": "not_applicable" if function["kind"] == "helper" else "review_required",
            "broadcasting": "not_applicable" if function["kind"] == "helper" else "review_required",
            "dtype": "review_required",
            "rank": "review_required",
            "cpu": "implemented",
            "wgpu": "review_required",
            "cuda": "review_required",
            "test_references": references,
            "has_visible_test_reference": bool(references),
        }
        operation_override = dict(overrides.get(name, {}))
        # Accept the original override spelling while emitting the canonical
        # matrix field used by REVIEW_FIELDS and the Markdown renderer.
        if "broadcast" in operation_override and "broadcasting" not in operation_override:
            operation_override["broadcasting"] = operation_override.pop("broadcast")
        entry.update(operation_override)
        entries.append(entry)

    operations = [entry for entry in entries if entry["kind"] == "operation"]
    reviewed = [
        entry
        for entry in operations
        if all(entry[field] != "review_required" for field in REVIEW_FIELDS)
    ]
    return {
        "schema_version": 1,
        "source": "src/tensor.rs",
        "annotation_source": "conformance/tensor_api_overrides.json",
        "summary": {
            "public_api_count": len(entries),
            "operation_count": len(operations),
            "helper_count": len(entries) - len(operations),
            "operations_with_visible_test_references": sum(
                bool(entry["has_visible_test_reference"]) for entry in operations
            ),
            "fully_reviewed_operations": len(reviewed),
        },
        "apis": entries,
    }


def render_markdown(matrix: dict[str, object]) -> str:
    summary = matrix["summary"]
    entries = matrix["apis"]
    rows = [
        "# Canonical Tensor API Conformance",
        "",
        "Generated by `python scripts/generate_tensor_conformance.py`.",
        "Edit `conformance/tensor_api_overrides.json` to record reviewed support.",
        "",
        f"- Public APIs: {summary['public_api_count']}",
        f"- Operations: {summary['operation_count']}",
        f"- Helpers/lifecycle APIs: {summary['helper_count']}",
        f"- Operations with visible test references: {summary['operations_with_visible_test_references']}",
        f"- Fully reviewed operations: {summary['fully_reviewed_operations']}",
        "",
        "| API | Kind | Tests | Backward | Broadcast | DType | Rank | CPU | WGPU | CUDA |",
        "|---|---|---:|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        rows.append(
            "| `{name}` | {kind} | {tests} | {backward} | {broadcasting} | "
            "{dtype} | {rank} | {cpu} | {wgpu} | {cuda} |".format(
                name=entry["name"],
                kind=entry["kind"],
                tests=len(entry["test_references"]),
                backward=entry["backward"],
                broadcasting=entry["broadcasting"],
                dtype=entry["dtype"],
                rank=entry["rank"],
                cpu=entry["cpu"],
                wgpu=entry["wgpu"],
                cuda=entry["cuda"],
            )
        )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    matrix = build_matrix()
    json_text = json.dumps(matrix, indent=2, sort_keys=False) + "\n"
    markdown_text = render_markdown(matrix)

    if args.check:
        stale = []
        if not JSON_OUTPUT.exists() or JSON_OUTPUT.read_text(encoding="utf-8") != json_text:
            stale.append(JSON_OUTPUT.relative_to(ROOT).as_posix())
        if (
            not MARKDOWN_OUTPUT.exists()
            or MARKDOWN_OUTPUT.read_text(encoding="utf-8") != markdown_text
        ):
            stale.append(MARKDOWN_OUTPUT.relative_to(ROOT).as_posix())
        if stale:
            print("Stale conformance artifacts: " + ", ".join(stale))
            return 1
        print("Tensor conformance artifacts are current.")
        return 0

    JSON_OUTPUT.write_text(json_text, encoding="utf-8")
    MARKDOWN_OUTPUT.write_text(markdown_text, encoding="utf-8")
    print(
        f"Generated {matrix['summary']['public_api_count']} API entries "
        f"({matrix['summary']['operation_count']} operations)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
