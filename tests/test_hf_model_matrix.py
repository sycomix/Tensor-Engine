"""Hugging Face model-matrix conformance.

Phase 0 gate: every model claim in `conformance/hf_model_matrix.json` is
explicit, version-pinned, and consistent with what the config detector in
`src/server/mod.rs` accepts. The server must reject unknown `model_type`
values instead of silently defaulting to Llama.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MATRIX = ROOT / "conformance" / "hf_model_matrix.json"
SERVER = ROOT / "src" / "server" / "mod.rs"
STATUSES = {"not_supported", "partial", "supported"}
REQUIRED_FIELDS = {
    "id",
    "architecture",
    "architectures",
    "upstream_config_version",
    "tokenizer_types",
    "supported_dtypes",
    "attention_variants",
    "rope",
    "tied_embeddings",
    "training_status",
    "inference_status",
    "fixture_ids",
}


def load_matrix() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def server_model_types() -> set[str]:
    text = SERVER.read_text(encoding="utf-8")
    match = re.search(r"fn detect_architecture\(&self\)[^{]*\{", text)
    assert match, "detect_architecture not found in src/server/mod.rs"
    body = text[match.end():]
    closing = body.index("}\n")
    body = body[:closing]
    return set(re.findall(r'"([^"]+)"', body))


def test_matrix_is_valid_json_schema() -> None:
    matrix = load_matrix()
    assert matrix["schema_version"] == 1
    assert isinstance(matrix["architectures"], list) and matrix["architectures"]
    seen_ids: set[str] = set()
    seen_model_types: set[str] = set()
    for entry in matrix["architectures"]:
        assert REQUIRED_FIELDS <= set(entry), f"entry {entry.get('id')} missing fields"
        assert entry["id"] not in seen_ids, f"duplicate id {entry['id']}"
        assert entry["architecture"] not in seen_model_types, (
            f"duplicate model_type {entry['architecture']}"
        )
        seen_ids.add(entry["id"])
        seen_model_types.add(entry["architecture"])
        assert entry["training_status"] in STATUSES
        assert entry["inference_status"] in STATUSES
        assert isinstance(entry["fixture_ids"], list)


def test_matrix_covers_phase_zero_families() -> None:
    matrix = load_matrix()
    model_types = {entry["architecture"] for entry in matrix["architectures"]}
    for required in ("llama", "mistral", "qwen2", "qwen3", "gemma2", "phi3"):
        assert required in model_types, f"matrix missing Phase 0 family {required}"


def test_matrix_claims_are_accepted_by_server_detector() -> None:
    matrix = load_matrix()
    accepted = server_model_types()
    for entry in matrix["architectures"]:
        assert entry["architecture"] in accepted, (
            f"matrix claims model_type {entry['architecture']!r} but the server "
            "config detector does not accept it"
        )


def test_server_rejects_unknown_model_type() -> None:
    text = SERVER.read_text(encoding="utf-8")
    assert "defaulting to Llama" not in text, (
        "server still silently falls back to Llama for unknown model_type"
    )
    assert "default_model_type" not in text, (
        "server still defaults a missing model_type to a fixed architecture"
    )
    assert "unsupported model_type" in text
