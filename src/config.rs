pub mod server {
    pub const DEFAULT_PORT: u16 = 8080;
    pub const DEFAULT_HOST: &str = "127.0.0.1";
    pub const DEFAULT_MAX_CONCURRENT_INFERENCES: usize = 5;
    pub const DEFAULT_PROMPT_CACHE_SIZE: usize = 50;
    pub const DEFAULT_API_PATH: &str = "/engine/v1/inference";
}

pub mod filenames {
    pub const CONFIG_JSON: &str = "config.json";
    pub const PARAMS_JSON: &str = "params.json";
    pub const MANIFEST_TXT: &str = "manifest.txt";
}

pub mod inference {
    pub const DEFAULT_TEMPERATURE: f32 = 1.0;
    pub const DEFAULT_TOP_P: f32 = 1.0;
    pub const DEFAULT_TOP_K: usize = 20;
    pub const DEFAULT_REPETITION_PENALTY: f32 = 1.0;
    pub const DEFAULT_MAX_SEQ_LEN: usize = 1024;
    pub const DEFAULT_MAX_NEW_TOKENS: usize = 20;
}

pub mod prompts {
    pub const DEFAULT_SYSTEM_PROMPT: &str = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, terse answers to the human's questions.### Human:";
    pub const DEFAULT_INTERACTIVE_PREFIX: &str = " ";
    pub const DEFAULT_INTERACTIVE_POSTFIX: &str = "### Assistant:";

    pub fn default_stop_tokens() -> Vec<String> {
        vec![
            "### Human:".to_string(),
            "###Human:".to_string(),
            "### Human: ".to_string(),
            "###Human: ".to_string(),
            " ### Human:".to_string(),
            " ###Human:".to_string(),
            " ### Human: ".to_string(),
            " ###Human: ".to_string(),
            "\n### Human:".to_string(),
            "\n###Human:".to_string(),
            "\n### Human: ".to_string(),
            "\n###Human: ".to_string(),
            "\n ### Human:".to_string(),
            "\n ###Human:".to_string(),
            "\n ### Human: ".to_string(),
            "\n ###Human: ".to_string(),
        ]
    }
}

pub mod opencl {
    pub const DEFAULT_DEVICE_IDX: usize = 0;
    pub const DEFAULT_GPU_PERCENTAGE: f32 = 1.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_defaults() {
        assert_eq!(server::DEFAULT_PORT, 8080);
        assert_eq!(server::DEFAULT_HOST, "127.0.0.1");
        assert!(server::DEFAULT_MAX_CONCURRENT_INFERENCES > 0);
    }

    #[test]
    fn test_filenames() {
        assert!(filenames::CONFIG_JSON.ends_with(".json"));
        assert!(filenames::PARAMS_JSON.ends_with(".json"));
    }

    #[test]
    fn test_inference_defaults() {
        assert!(inference::DEFAULT_TEMPERATURE > 0.0);
        assert!(inference::DEFAULT_TOP_K > 0);
    }

    #[test]
    fn test_prompts() {
        assert!(!prompts::DEFAULT_SYSTEM_PROMPT.is_empty());
        let stops = prompts::default_stop_tokens();
        assert!(!stops.is_empty());
        assert!(stops.contains(&"### Human:".to_string()));
    }
}
