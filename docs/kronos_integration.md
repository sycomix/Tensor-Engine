# Kronos Format Integration

This doc explains how the `kronos` format is supported in `tensor_engine`.

- The Kronos format is encoded in SafeTensors archives and includes a metadata marker `__kronos_marker__` with value `4B524F4E` (hex for 'KRON') or `format: Kronos`.
- Use `tensor_engine::io::safetensors_loader::apply_kronos_bytes_to_module_bytes` to load a Kronos file into a `MultimodalLLM` instance.
- The function maps Kronos keys such as `vision_encoder.*`, `text_embedding.weight`, `projector.*`, `decoder_blocks.layers.*`, and `head.*` into the module fields.

See `kronos-modal-format.md` for the full schema and recommended mapping semantics.