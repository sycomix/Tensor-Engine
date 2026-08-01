use ndarray::{ArrayD, IxDyn};
use tensor_engine::dtype::DType;
use tensor_engine::quantization::awq::unpack_4bit_u8;
use tensor_engine::tensor::Tensor;

#[test]
fn test_unpack_4bit_u8_simple() {
    // 1. Create a packed U8 tensor.
    // We want to represent a 4x4 matrix of f32s (16 elements).
    // In 4-bit, this takes 16 * 4 bits = 64 bits = 8 bytes.
    // Let's manually construct the bytes.
    // Values: 0..15
    // Byte 0: [0, 1] -> 0x10 (if high is at bits 4-7, low at 0-3) or 0x01?
    // unpack_4bit_u8 logic:
    //   low = byte & 0x0F
    //   high = (byte >> 4) & 0x0F
    // So if we want vals[0]=0, vals[1]=1:
    //   byte = (1 << 4) | 0 = 16 = 0x10.

    // Let's encode values 0..15.
    let mut bytes: Vec<u8> = Vec::new();
    for i in 0..8 {
        let low = i * 2; // 0, 2, 4, up to 14
        let high = i * 2 + 1; // 1, 3, 5, up to 15
        let b = (low & 0x0F) | ((high & 0x0F) << 4);
        bytes.push(b as u8);
    }

    // Create Tensor.
    // We can't use Tensor::new() effectively for U8 unless we rely on raw storage manipulation
    // OR we use the fact that I enabled DType::U8 in from_f32_array but that casts f32->u8.
    // So let's create an f32 array of these byte values, then astype(U8).
    let byte_floats: Vec<f32> = bytes.iter().map(|&b| b as f32).collect();
    let shape_packed = [8];
    let t_packed_f32 = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape_packed[..]), byte_floats).unwrap(),
        false,
    );

    let t_packed = t_packed_f32.astype(DType::U8);

    // Verify storage is indeed U8
    assert_eq!(t_packed.dtype(), DType::U8);

    // 2. Unpack
    let target_shape = vec![4, 4]; // 16 elements
    let t_unpacked = unpack_4bit_u8(&t_packed, &target_shape).expect("unpack failed");

    assert_eq!(t_unpacked.dtype(), DType::F32);
    let out_data = t_unpacked.to_f32_array();

    assert_eq!(out_data.shape(), &[4, 4]);

    // Verify values
    let flat_out: Vec<f32> = out_data.iter().cloned().collect();
    for i in 0..16 {
        assert_eq!(flat_out[i], i as f32, "Mismatch at index {}", i);
    }
}

#[test]
fn test_unpack_4bit_u8_shape_check() {
    let bytes = [0u8; 4]; // 8 nibbles
    let byte_floats: Vec<f32> = bytes.iter().map(|&b| b as f32).collect();
    let t_packed = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4][..]), byte_floats).unwrap(),
        false,
    )
    .astype(DType::U8);

    // unpacking into 9 elements should fail (panic in current impl)
    // We catch unwind to verify panic
    // unpacking into 9 elements should fail (return Err)
    let result = unpack_4bit_u8(&t_packed, &[9][..]);
    assert!(
        result.is_err(),
        "Expected error for 9 elements (shape mismatch)"
    );

    // unpacking into 8 elements should succeed
    let result_ok = unpack_4bit_u8(&t_packed, &[8][..]);
    assert!(result_ok.is_ok(), "Expected success for 8 elements");
}

#[test]
fn test_awq_dequantize_affine_simple() {
    // Test manual dequantize function: w = (q - z) * s
    // Shape: [4, 4], Group size 2
    // Rows: 4, Cols: 4
    // Packed: [4, 2] U8 (16 elements)
    // Scales: [4, 2] F32 (groups=4/2=2)
    // Zeros: [4, 2] F32

    use tensor_engine::quantization::awq::awq_dequantize_affine;

    // Create packed weights: all 0x55 (nibbles 5 and 5)
    // 16 elements of value 5.
    let packed_data = [0x55u8; 8];
    let packed = Tensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[8][..]),
            packed_data.iter().map(|&x| x as f32).collect(),
        )
        .unwrap(),
        false,
    )
    .astype(DType::U8);

    // Scales: all 0.5
    // Shape [4, 2]
    let scales = Tensor::new(ArrayD::from_elem(IxDyn(&[4, 2][..]), 0.5f32), false);

    // Zeros: all 1.0 (float)
    // Shape [4, 2]
    // w = (5 - 1) * 0.5 = 4 * 0.5 = 2.0
    let zeros = Tensor::new(ArrayD::from_elem(IxDyn(&[4, 2][..]), 1.0f32), false);

    let out =
        awq_dequantize_affine(&packed, &scales, &zeros, 2, &[4, 4]).expect("dequant failed");

    let out_data = out.to_f32_array();
    assert_eq!(out_data.shape(), &[4, 4]);
    for v in out_data.iter() {
        assert!((v - 2.0).abs() < 1e-5, "Expected 2.0, got {}", v);
    }
}

#[test]
fn test_quantized_linear_module() {
    use tensor_engine::nn::quantized::QuantizedLinear;
    use tensor_engine::nn::Module;

    // In=4, Out=4, Group=4 (1 group per row)
    // Weight = 3.0 everywhere
    // q=4, z=1, s=1.0 -> (4-1)*1 = 3.0

    // Packed q=4 -> 0x44
    let packed_data = [0x44u8; 8]; // 16 elts
    let qweight = Tensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[8][..]),
            packed_data.iter().map(|&x| x as f32).collect(),
        )
        .unwrap(),
        false,
    )
    .astype(DType::U8);

    // Shape [4, 1] for scales/zeros (since group_size=4, cols=4, so 1 group)
    let scales = Tensor::new(ArrayD::from_elem(IxDyn(&[4, 1][..]), 1.0f32), false);
    let qzeros = Tensor::new(ArrayD::from_elem(IxDyn(&[4, 1][..]), 1.0f32), false);
    let bias = Tensor::new(ArrayD::from_elem(IxDyn(&[4][..]), 0.5f32), false);

    let layer = QuantizedLinear::new(qweight, qzeros, scales, Some(bias), 4, 4, 4);

    // Input: Identity 4x4
    let eye = ArrayD::from_shape_vec(
        IxDyn(&[4, 4][..]),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    let input = Tensor::new(eye, false);

    // Forward
    // If weights are all 3.0, W is 4x4 of 3.0s.
    // X @ W + b
    // I @ W = W
    // W + b:
    // Row 0: [3, 3, 3, 3] + 0.5? Broadcasting of bias usually adds to last dim.
    // bias is [4].
    // W is [4, 4]. W + b -> Each row gets +b? Or each column?
    // In pytorch Linear(in, out), weight is [out, in]. x @ w.T + b.
    // Here we assumed qweight is [in, out] in our QuantizedLinear logic (check implementation).
    // Our impl:
    //   target_shape = vec![self.in_features, self.out_features];
    //   w = unpack(packed_4bit_values)
    //   input.matmul(&w)
    // MatMul(A, B): A[m, k] @ B[k, n] -> C[m, n].
    // Input [4, 4], W [4, 4]. Out [4, 4].
    // Bias add: Standard broadcast usually adds to last dim.
    // If bias is [4], and out is [4, 4] -> adds to each row.
    // So expected output row = [3.5, 3.5, 3.5, 3.5]

    let out = layer.forward(&input);
    let out_arr = out.to_f32_array();

    for v in out_arr.iter() {
        assert!((v - 3.5).abs() < 1e-5, "Expected 3.5, got {}", v);
    }
}
