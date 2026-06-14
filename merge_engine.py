"import os"  
""  
'with open(r"D:\\tensor-engine\\src\\bin\\llm.rs", "r") as f:'  
'    llm_content = f.read()'  
""  
'footer = """' >> D:\tensor-engine\merge_engine.py && echo '#[cfg(feature = "compat")]' >> D:\tensor-engine\merge_engine.py && echo 'mod compat_engine {' >> D:\tensor-engine\merge_engine.py && echo '    pub use tensor_engine::compat::engine::entrypoint;' >> D:\tensor-engine\merge_engine.py && echo '}' >> D:\tensor-engine\merge_engine.py && echo '' >> D:\tensor-engine\merge_engine.py && echo 'fn main() {' >> D:\tensor-engine\merge_engine.py && echo '    let args: Vec<String> = env::args().collect();' >> D:\tensor-engine\merge_engine.py && echo '' >> D:\tensor-engine\merge_engine.py && echo '    if args.len() >= 2 && args[1] == "compat" {' >> D:\tensor-engine\merge_engine.py && echo '        #[cfg(feature = "compat")]' >> D:\tensor-engine\merge_engine.py && echo '        {' >> D:\tensor-engine\merge_engine.py && echo '            if let Err(e) = compat_engine::entrypoint::main() {' >> D:\tensor-engine\merge_engine.py && echo '                eprintln!("Error: {}", e);' >> D:\tensor-engine\merge_engine.py && echo '                std::process::exit(1);' >> D:\tensor-engine\merge_engine.py && echo '            }' >> D:\tensor-engine\merge_engine.py && echo '            return;' >> D:\tensor-engine\merge_engine.py && echo '        }' >> D:\tensor-engine\merge_engine.py && echo '        #[cfg(not(feature = "compat"))]' >> D:\tensor-engine\merge_engine.py && echo '        {' >> D:\tensor-engine\merge_engine.py && echo '            eprintln!("The compat subcommand requires the compat feature.");' >> D:\tensor-engine\merge_engine.py && echo '            eprintln!("Run with: cargo run --bin engine --features compat");' >> D:\tensor-engine\merge_engine.py && echo '            std::process::exit(1);' >> D:\tensor-engine\merge_engine.py && echo '        }' >> D:\tensor-engine\merge_engine.py && echo '    }' >> D:\tensor-engine\merge_engine.py && echo '' >> D:\tensor-engine\merge_engine.py && echo '    if let Err(err) = run() {' >> D:\tensor-engine\merge_engine.py && echo '        eprintln!("error: {}", err);' >> D:\tensor-engine\merge_engine.py && echo '        std::process::exit(1);' >> D:\tensor-engine\merge_engine.py && echo '    }' >> D:\tensor-engine\merge_engine.py && echo '}' >> D:\tensor-engine\merge_engine.py && echo '"""'  
""  
'with open(r"D:\\tensor-engine\\src\\bin\\engine.rs", "w") as f:'  
'    f.write("// Tensor Engine unified binary - merges llm commands + compat subcommand\\n")'  
'    f.write(llm_content)'  
'    f.write(footer)'  
""  
'import os'  
'size = os.path.getsize(r"D:\\tensor-engine\\src\\bin\\engine.rs")'  
'print(f"Wrote {size} bytes to engine.rs")' 
