use std::process::Command;

#[test]
fn verify_as_any_mut_script_runs() {
    // If 'python' isn't available, skip the test (many CI runners have python, but be tolerant locally)
    let python = if cfg!(target_os = "windows") { "python" } else { "python3" };
    match Command::new(python).arg("scripts/verify_as_any_mut.py").status() {
        Ok(s) => {
            if !s.success() {
                panic!("scripts/verify_as_any_mut.py failed with status: {}", s);
            }
        }
        Err(e) => {
            eprintln!("python not found or failed to run: {}. Skipping as_any_mut verification test.", e);
        }
    }
}
