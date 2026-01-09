extern crate proc_macro;

use proc_macro2::TokenStream;
use quote::quote;
use syn;

mod attr;
mod call_fn;
mod call_impl;
mod ext;
mod function;
mod ptr_type;
mod return_type;

use attr::invoke::InvokeParams;
use crate::ext::ErrorExt;

#[proc_macro_attribute]
pub fn marshal(
    params: proc_macro::TokenStream,
    function: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    // Parse attribute args using syn to be resilient across darling versions.
    // Parse attribute arguments into a Vec of NestedMeta using the compiler-friendly helper macro.
    let parsed_args = syn::parse_macro_input!(params as syn::AttributeArgs);
    // Translate syn::AttributeArgs into InvokeParams (a small, explicit conversion)
    let mut invoke = InvokeParams::default();
    for nested in parsed_args.iter() {
        match nested {
            syn::NestedMeta::Meta(syn::Meta::Path(path)) => {
                if path.is_ident("callback") {
                    invoke.callback = true;
                } else {
                    // treat a bare path as a return_marshaler path
                    invoke.return_marshaler = Some(path.clone());
                }
            }
            syn::NestedMeta::Meta(syn::Meta::NameValue(nv)) => {
                if nv.path.is_ident("prefix") {
                    if let syn::Lit::Str(s) = &nv.lit {
                        invoke.prefix = Some(s.value());
                    }
                } else if nv.path.is_ident("return_marshaler") {
                    if let syn::Lit::Str(s) = &nv.lit {
                        if let Ok(p) = syn::parse_str::<syn::Path>(&s.value()) {
                            invoke.return_marshaler = Some(p);
                        }
                    }
                } else if nv.path.is_ident("callback") {
                    if let syn::Lit::Bool(b) = &nv.lit {
                        invoke.callback = b.value;
                    }
                }
            }
            _ => {}
        }
    }

    match call_with(invoke, function.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => proc_macro::TokenStream::from(syn::Error::new(err.span(), err.to_string()).to_compile_error()),
    }
}

use std::sync::Once;
static INIT_LOGGER: Once = Once::new();
fn ensure_logger() {
    INIT_LOGGER.call_once(|| { let _ = pretty_env_logger::try_init(); });
}

fn call_with(invoke_params: InvokeParams, item: TokenStream) -> Result<TokenStream, syn::Error> {
    // Ensure logging is initialized lazily to avoid #[ctor] / linker issues on MSVC.
    ensure_logger();
    let item: syn::Item = syn::parse2(item.clone()).context("error parsing function body")?;
    let result = match item {
        syn::Item::Fn(item) => call_fn::call_with_function(invoke_params.return_marshaler, invoke_params.callback, item, None),
        syn::Item::Impl(item) => call_impl::call_with_impl(invoke_params.prefix, item),
        item => {
            log::error!("{}", quote! { #item });
            Err(syn::Error::new_spanned(&item, "Only supported on functions and impls"))
        }
    };

    if result.is_err() {
        log::debug!("macro finished with error");
    } else {
        log::debug!("macro finished successfully");
    }

    result
}

include!(concat!(env!("OUT_DIR"), "/codegen.rs"));

pub(crate) fn default_marshaler(ty: &syn::Type) -> Option<syn::Path> {
    DEFAULT_MARSHALERS
        .get(&*quote! { #ty }.to_string())
        .and_then(|x| syn::parse_str(x).ok())
}

pub(crate) fn is_passthrough_type(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::BareFn(bare_fn) => bare_fn.abi.is_some(),
        _ => PASSTHROUGH_TYPES.contains(&&*quote! { #ty }.to_string()),
    }
}