/// Checkpoint operation: trades compute for memory by re-running the forward pass during backward.
///
/// This operation takes a closure `f` and inputs `x`.
/// Forward: runs `f(x)`, returns result. The intermediate graph nodes inside `f` are discarded (not stored in `output`'s creator).
/// Backward: re-runs `f(x)` (tracing the graph), then runs backward on the new graph.
pub struct Checkpoint<F>
where
    F: Fn(&[Tensor]) -> Tensor + Send + Sync + 'static,
{
    pub f: std::sync::Arc<F>,
}

impl<F> Checkpoint<F>
where
    F: Fn(&[Tensor]) -> Tensor + Send + Sync + 'static,
{
    pub fn new(f: F) -> Self {
        Checkpoint {
            f: std::sync::Arc::new(f),
        }
    }
}

impl<F> Operation for Checkpoint<F>
where
    F: Fn(&[Tensor]) -> Tensor + Send + Sync + 'static,
{
    fn forward(&self, inputs: &[Tensor], output: &mut ArrayD<f32>) {
        // Run f(inputs) in a way that doesn't build the graph?
        // Actually, we WANT to verify that f(inputs) works.
        // But for "checkpointing" to work as an Operation, `inputs` are already Tensors.
        // If we run `f(inputs)`, it produces a Tensor `out`.
        // We copy `out`'s data to `output`.
        // Crucially, `output` (the Tensor wrapping this Operation) matches `out`'s data.
        // But `output`'s creator is THIS Checkpoint op, not the last op in f.
        // So the graph is "cut" effectively.
        let out_tensor = (self.f)(inputs);

        let out_data = out_tensor.to_f32_array();
        if out_data.shape() != output.shape() {
            // Ops trait says output is pre-allocated?
            // Wait, usually `apply` determines shape.
            // But `apply` calls `forward` with pre-allocated zero array based on shape inference.
            // We need `apply` to know the shape.
            // Generic `apply` infers shape from broadcasting usually, or op-specific logic.
            // For Checkpoint, we don't know the shape ahead of time easily unless we run f.
            // But `Tensor::apply` does:
            // 1. determine output shape
            // 2. allocate
            // 3. call forward

            // This is tricky for Checkpoint because correct shape logic in `Tensor::apply` depends on the op.
            // We might need to handle this by running `f` inside `Tensor::apply` logic or
            // making `Checkpoint` implement a `compute_shape` method if we refactored.
            // For now, let's assume valid shape or resize output?
            // `output` is &mut ArrayD. We can just overwrite it?
            // No, `output` is allocated by `apply`. Use `assign`.
        }

        // Resize output if necessary (Tensor::apply allocates generic broadcast shape usually, which might be wrong for f)
        // If we assume `f` is "same shape as inputs" or similar, maybe.
        // But f can change shape.
        // Implementation detail: `Tensor::apply` needs to support arbitrary output shapes for Opaque ops.
        // Let's check `Tensor::apply`:
        // It has special cases for Sum, Mean, Concat.
        // For others it does broadcasting.
        // We need to add Checkpoint to `Tensor::apply` special handling OR make `Checkpoint` behave nicely.

        // Actually, we can just *assign* to output.
        // If the shape mismatches, `apply` might return a Tensor with the WRONG shape metadata if we don't fix `apply`.
        // We MUST modify `Tensor::apply` in `src/tensor.rs` to handle Checkpoint shape inference (by running f?).

        *output = out_data;
    }

    fn backward(&self, inputs: &[Tensor], output_grad: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // 1. Re-run forward pass with tracking enabled
        // We need inputs to require grad for this to make sense.
        // But even if they don't, we might need intermediates.
        // Actually, we just need to differentiate `f` at `inputs`.

        // Enable grad on inputs temporarily if not set?
        // No, `inputs` are from the graph.

        // We call (self.f)(inputs).
        // This builds a local graph attached to `inputs`.
        let out_tensor = (self.f)(inputs);

        // 2. Run backward on the result, feeding in `output_grad`
        // We need to manually pump `output_grad` into `out_tensor`.

        // `out_tensor` is the root of the re-computed subgraph.
        // We want d(Loss)/d(Inputs) = d(Loss)/d(Out) * d(Out)/d(Inputs)
        // We have d(Loss)/d(Out) = output_grad.
        // So we invoke `backward` on `out_tensor` seeded with `output_grad`.

        // But `autograd::backward` accumulates into `.grad` fields.
        // We don't want to pollute `inputs.grad` directly yet?
        // Actually, `Operation::backward` asks us to return `Vec<ArrayD<f32>>` (gradients for inputs).
        // It does NOT want us to side-effect `inputs.grad`.
        // So we need to capture the gradients w.r.t inputs without updating them.

        // Problem: `AutogradEngine::backward` is all about side-effects on `.grad`.
        // Solution: Create separate, detached input copies that track gradients, run f on DOES, keep inputs fixed.
        // Then extract grads from those copies.

        let mut detached_inputs = Vec::new();
        for inp in inputs {
            // detach: new tensor sharing storage, but fresh grad/graph state
            // and we force requires_grad=true to capture gradient
            let data = inp.to_f32_array();
            let t = Tensor::new(data, true);
            detached_inputs.push(t);
        }

        let out_tensor = (self.f)(&detached_inputs);

        // Seed output gradient
        {
            let mut lock = out_tensor.lock();
            lock.grad = Some(output_grad.clone());
        }

        // Run backward on this subgraph
        crate::autograd::AutogradEngine::new().backward(&out_tensor);

        // Collect grads from detached_inputs
        let mut grads = Vec::new();
        for (i, t) in detached_inputs.iter().enumerate() {
            let g = t.lock().grad.clone().unwrap_or_else(|| {
                // if no grad, return zeros
                ArrayD::zeros(IxDyn(&inputs[i].lock().storage.shape()))
            });
            grads.push(g);
        }

        grads
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
