#[derive(Default)]
pub struct InvokeParams {
    pub return_marshaler: Option<syn::Path>,
    pub prefix: Option<String>,
    pub callback: bool,
}
