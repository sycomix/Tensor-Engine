use tensor_engine::nn::RVQ;
use tensor_engine::tensor::Tensor;
use ndarray::IxDyn;

#[test]
fn test_rvq_update_ema_simple() {
    // single level, 2 codes, dim 1
    let mut rvq = RVQ::new(2, 1, 1);
    // initialize codebook to 0 and 10
    let mut cb0 = rvq.codebooks[0].lock();
    cb0.storage = tensor_engine::dtype::TensorStorage::from_f32_array(&ndarray::arr2(&[[0.0f32], [10.0f32]]).into_dyn(), cb0.dtype);
    drop(cb0);

    // inputs: 1 assigned to code 0, 9 assigned to code 1
    let arr = ndarray::Array::from_shape_vec(IxDyn(&[2, 1]), vec![1.0f32, 9.0f32]).unwrap();
    let x = Tensor::new(arr.into_dyn(), false);
    let indices = rvq.quantize(&x);
    assert_eq!(indices.len(), 1);
    let idxs = &indices[0];
    assert_eq!(idxs.len(), 2);
    // update ema with decay 0.0 to force instant mean
    rvq.update_ema(&x, &indices, 0.0).expect("update failed");
    let cb_after = rvq.codebooks[0].lock().storage.to_f32_array();
    assert_eq!(cb_after[[0,0]], 1.0f32);
    assert_eq!(cb_after[[1,0]], 9.0f32);
}

#[test]
fn test_rvq_update_ema_schedule() {
    let mut rvq = RVQ::new(2, 1, 1);
    let mut cb0 = rvq.codebooks[0].lock();
    cb0.storage = tensor_engine::dtype::TensorStorage::from_f32_array(&ndarray::arr2(&[[0.0f32], [10.0f32]]).into_dyn(), cb0.dtype);
    drop(cb0);
    let arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[2, 1]), vec![1.0f32, 9.0f32]).unwrap();
    let x = Tensor::new(arr.into_dyn(), false);
    let indices = rvq.quantize(&x);
    // schedule updates to every 2 calls
    rvq.set_ema_update_every(2);
    // call once: should not update
    rvq.update_ema(&x, &indices, 0.0).expect("update failed");
    let cb_after1 = rvq.codebooks[0].lock().storage.to_f32_array();
    assert_eq!(cb_after1[[0,0]], 0.0f32);
    assert_eq!(cb_after1[[1,0]], 10.0f32);
    // call again: should update
    rvq.update_ema(&x, &indices, 0.0).expect("update failed");
    let cb_after2 = rvq.codebooks[0].lock().storage.to_f32_array();
    assert_eq!(cb_after2[[0,0]], 1.0f32);
    assert_eq!(cb_after2[[1,0]], 9.0f32);
}

#[test]
fn test_rvq_update_ema_reinit_empty() {
    let mut rvq = RVQ::new(2, 1, 1);
    let mut cb0 = rvq.codebooks[0].lock();
    cb0.storage = tensor_engine::dtype::TensorStorage::from_f32_array(&ndarray::arr2(&[[0.0f32], [100.0f32]]).into_dyn(), cb0.dtype);
    drop(cb0);
    let arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[2, 1]), vec![1.0f32, 1.0f32]).unwrap();
    let x = Tensor::new(arr.into_dyn(), false);
    let indices = rvq.quantize(&x);
    // reinit empty codes
    rvq.set_reinit_empty_codes(true);
    rvq.update_ema(&x, &indices, 0.5).expect("update failed");
    let cb_after = rvq.codebooks[0].lock().storage.to_f32_array();
    // the second entry should no longer be 100.0 because it was reinitialized
    assert!(cb_after[[1,0]] != 100.0f32);
}
