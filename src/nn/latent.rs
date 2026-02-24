use crate::tensor::Tensor;

/// Utilities for operating on latent vectors in the joint embedding space.
///
/// All functions work on arbitrarily-shaped tensors as long as the shapes match
/// in the call.  They are written to be safe for production use: inputs are
/// validated and errors are logged rather than panicking, and the output is a
/// brand‑new `Tensor` so callers can treat them as pure functions.  Some of the
/// operations (notably `spherical_interpolate`) perform their maths on the
/// underlying `ndarray` data and therefore do not participate in automatic
/// differentiation.  This is intentional: interpolation is typically used at
/// inference time or within analysis tools where gradients are not needed.

/// Elementwise arithmetic on latent vectors: A - B + C.
pub fn vector_arithmetic(a: &Tensor, b: &Tensor, c: &Tensor) -> Tensor {
    a.sub(b).add(c)
}

/// Linear interpolation between two latent points.
///
/// Computes `(1 - alpha) * p + alpha * q` with broadcasting support.  `alpha`
/// should be in the range `[0,1]` but the routine will happily accept other
/// values and behave accordingly (extrapolation).
pub fn linear_interpolate(p: &Tensor, q: &Tensor, alpha: f32) -> Tensor {
    let one_minus = Tensor::new(ndarray::arr0(1.0 - alpha).into_dyn(), false);
    let alpha_t = Tensor::new(ndarray::arr0(alpha).into_dyn(), false);
    p.mul(&one_minus).add(&q.mul(&alpha_t))
}

/// Spherical linear interpolation (slerp) between two latent points.
///
/// This routine first normalizes `p` and `q`, computes the angle between them,
/// and then interpolates along the great circle.  The implementation works on
/// the underlying `ndarray` data and returns a non‑grad‑tracked tensor; this is
/// acceptable for typical usage where the interpolation itself is not part of a
/// training objective.
pub fn spherical_interpolate(p: &Tensor, q: &Tensor, alpha: f32) -> Tensor {
    // Extract arrays so we can do some scalar maths that the core tensor API
    // does not provide (e.g. acos, sin).
    let p_arr = p.lock().storage.to_f32_array();
    let q_arr = q.lock().storage.to_f32_array();
    if p_arr.len() != q_arr.len() {
        log::error!(
            "spherical_interpolate: shape mismatch ({} vs {}), falling back to linear",
            p_arr.len(),
            q_arr.len()
        );
        return linear_interpolate(p, q, alpha);
    }
    // compute dot product and norms
    let mut dot = 0.0f32;
    let mut norm_p_sq = 0.0f32;
    let mut norm_q_sq = 0.0f32;
    for (x, y) in p_arr.iter().zip(q_arr.iter()) {
        dot += x * y;
        norm_p_sq += x * x;
        norm_q_sq += y * y;
    }
    let norm_p = norm_p_sq.sqrt();
    let norm_q = norm_q_sq.sqrt();
    if norm_p == 0.0 || norm_q == 0.0 {
        // degenerate input, just linear interp to avoid divide-by-zero
        return linear_interpolate(p, q, alpha);
    }
    let mut cos_theta = dot / (norm_p * norm_q);
    if cos_theta > 1.0 {
        cos_theta = 1.0;
    } else if cos_theta < -1.0 {
        cos_theta = -1.0;
    }
    let theta = cos_theta.acos();
    if theta.abs() < 1e-6 {
        // points almost identical
        return p.clone();
    }
    let sin_theta = theta.sin();
    let s1 = ((1.0 - alpha) * theta).sin() / sin_theta;
    let s2 = (alpha * theta).sin() / sin_theta;

    let s1_t = Tensor::new(ndarray::arr0(s1).into_dyn(), false);
    let s2_t = Tensor::new(ndarray::arr0(s2).into_dyn(), false);
    p.mul(&s1_t).add(&q.mul(&s2_t))
}

/// Attribute-guided editing: move `base` in the direction of `attr` by a given
/// `strength` scalar.
pub fn attribute_edit(base: &Tensor, attr: &Tensor, strength: f32) -> Tensor {
    let s = Tensor::new(ndarray::arr0(strength).into_dyn(), false);
    base.add(&attr.mul(&s))
}
