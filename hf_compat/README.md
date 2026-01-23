# hf_compat

A small compatibility crate to load HuggingFace-style model bundles and unpickle PyTorch `data.pkl` entries.

This crate is optional and compiled when the root crate is built with `--features hf_compat`.

Usage (from root crate):

cargo build --features hf_compat

Then call `hf_compat::HugginfaceModel::unpickle(path)` to inspect model contents.
