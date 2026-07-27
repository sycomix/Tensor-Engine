#!/usr/bin/env python3
"""Enforce isolation between the canonical and legacy Tensor runtimes."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

LEGACY_DEFAULT_FEATURES = {"hf_compat", "opencl"}
GPT_FRAMEWORK_PREFIX = "src/nn/gpt/framework/"
GPT_FRAMEWORK_REFERENCE_ALLOW_PREFIXES = (
    GPT_FRAMEWORK_PREFIX,
    "src/nn/gpt/training/",
)


def default_features() -> set[str]:
    manifest = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r"^default\s*=\s*\[(.*?)\]", manifest, re.MULTILINE | re.DOTALL)
    if not match:
        raise RuntimeError("Cargo.toml has no default feature list")
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def rust_sources():
    yield from SRC.rglob("*.rs")


def main() -> int:
    violations: list[str] = []

    enabled_legacy = sorted(default_features() & LEGACY_DEFAULT_FEATURES)
    if enabled_legacy:
        violations.append(
            "Cargo default features enable legacy runtimes: "
            + ", ".join(enabled_legacy)
        )

    for path in rust_sources():
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")

        if re.search(r"(?:crate|tensor_engine)::compat\b", text):
            violations.append(f"{relative}: imports the removed compatibility runtime")

        if not relative.startswith(GPT_FRAMEWORK_REFERENCE_ALLOW_PREFIXES):
            if re.search(
                r"(?:crate::)?nn::gpt::framework\b|super::gpt::framework\b", text
            ):
                violations.append(f"{relative}: imports the GPT-specific runtime")

    bindings = (SRC / "python_bindings.rs").read_text(encoding="utf-8", errors="ignore")
    if not re.search(
        r"pub struct PyTensor\s*\{[^}]*\binner:\s*crate::tensor::Tensor",
        bindings,
        re.DOTALL,
    ):
        violations.append("src/python_bindings.rs: does not bind the canonical Tensor")

    if (SRC / "compat").exists():
        violations.append("src/compat: removed compatibility runtime exists")

    manifest = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    if re.search(r"^\s*compat\s*=", manifest, re.MULTILINE):
        violations.append("Cargo.toml: removed compat feature exists")

    engine_binary = (SRC / "bin" / "engine.rs").read_text(
        encoding="utf-8", errors="ignore"
    )
    if re.search(r"tensor_engine::compat\b", engine_binary):
        violations.append(
            "src/bin/engine.rs: canonical CLI delegates to the compatibility runtime"
        )
    if not re.search(r"tensor_engine::server::serve\s*\(", engine_binary):
        violations.append(
            "src/bin/engine.rs: serve command does not launch the canonical server"
        )

    gpt_module = (SRC / "nn/gpt/mod.rs").read_text(encoding="utf-8", errors="ignore")
    if "pub(crate) mod framework;" in gpt_module:
        violations.append(
            "src/nn/gpt/mod.rs: legacy framework still exists as crate-private module"
        )

    legacy_framework = SRC / "nn/gpt/framework"
    if legacy_framework.exists():
        violations.append(
            "src/nn/gpt/framework/: legacy framework directory still exists"
        )

    legacy_autograd = SRC / "nn/gpt/framework/autograd.rs"
    if legacy_autograd.exists():
        violations.append(
            "src/nn/gpt/framework/autograd.rs: duplicate Tensor/autograd implementation exists"
        )

    if violations:
        print("Canonical runtime boundary violations:")
        for violation in violations:
            print(f"  {violation}")
        return 1

    print("Canonical runtime boundaries passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
