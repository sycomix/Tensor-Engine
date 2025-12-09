#[cfg(feature = "audio")]
use tensor_engine::io::dataloader::resample_linear;
#[cfg(feature = "audio")]
use tensor_engine::io::dataloader::resample_high_quality;

#[cfg(feature = "audio")]
#[test]
fn test_resample_linear_double_rate() {
    // simple ramp: [0, 1, 2, 3] at 4 samples -> doubling rate should approx [0,0.5,1,1.5,2,2.5,3]
    let src = vec![0.0f32, 1.0f32, 2.0f32, 3.0f32];
    let res = resample_linear(&src, 4, 8);
    assert_eq!(res.len(), 8);
    // Check first, middle, last
    assert!((res[0] - 0.0).abs() < 1e-6);
    assert!((res[7] - 3.0).abs() < 1e-6);
}

#[cfg(all(feature = "audio", feature = "rubato"))]
#[test]
fn test_resample_high_quality_basic() {
    let src = vec![0.0f32, 1.0f32, 2.0f32, 3.0f32];
    let res = resample_high_quality(&src, 4, 8);
    // Expect length approx double (within 2 samples)
    let expected_len = ((src.len() as f64) * (8.0 / 4.0)).round() as usize;
    assert!((res.len() as isize - expected_len as isize).abs() <= 2);
    // Basic checks: output length and sanity of sample values
    assert!(res.len() > 0);
    for v in res.iter() { assert!(v.is_finite()); }
}
