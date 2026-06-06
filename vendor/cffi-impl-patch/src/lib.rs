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

use crate::ext::ErrorExt;
use attr::invoke::InvokeParams;

#[proc_macro_attribute]
pub fn marshal(
    params: proc_macro::TokenStream,
    function: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let parser = syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated;
    let parsed_args = syn::parse_macro_input!(params with parser);
    // Translate syn::AttributeArgs into InvokeParams (a small, explicit conversion)
    let mut invoke = InvokeParams::default();
    for meta in parsed_args.iter() {
        match meta {
            syn::Meta::Path(path) => {
                if path.is_ident("callback") {
                    invoke.callback = true;
                } else {
                    // treat a bare path as a return_marshaler path
                    invoke.return_marshaler = Some(path.clone());
                }
            }
            syn::Meta::NameValue(nv) => {
                if nv.path.is_ident("prefix") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(s),
                        ..
                    }) = &nv.value
                    {
                        invoke.prefix = Some(s.value());
                    }
                } else if nv.path.is_ident("return_marshaler") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(s),
                        ..
                    }) = &nv.value
                    {
                        if let Ok(p) = syn::parse_str::<syn::Path>(&s.value()) {
                            invoke.return_marshaler = Some(p);
                        }
                    }
                } else if nv.path.is_ident("callback") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Bool(b),
                        ..
                    }) = &nv.value
                    {
                        invoke.callback = b.value;
                    }
                }
            }
            _ => {}
        }
    }

    match call_with(invoke, function.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => proc_macro::TokenStream::from(
            syn::Error::new(err.span(), err.to_string()).to_compile_error(),
        ),
    }
}

use std::sync::Once;
static INIT_LOGGER: Once = Once::new();
fn ensure_logger() {
    INIT_LOGGER.call_once(|| {
        let _ = pretty_env_logger::try_init();
    });
}

fn call_with(invoke_params: InvokeParams, item: TokenStream) -> Result<TokenStream, syn::Error> {
    // Ensure logging is initialized lazily to avoid #[ctor] / linker issues on MSVC.
    ensure_logger();
    let item: syn::Item = syn::parse2(item.clone()).context("error parsing function body")?;
    let result = match item {
        syn::Item::Fn(item) => call_fn::call_with_function(
            invoke_params.return_marshaler,
            invoke_params.callback,
            item,
            None,
        ),
        syn::Item::Impl(item) => call_impl::call_with_impl(invoke_params.prefix, item),
        item => {
            log::error!("{}", quote! { #item });
            Err(syn::Error::new_spanned(
                &item,
                "Only supported on functions and impls",
            ))
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

#[allow(dead_code)]
use std::collections::HashMap;

#[allow(unused, unexpected_cfgs)]
fn get_default_marshalers() -> HashMap<String, String> {
    let mut map = HashMap::new();
    if cfg!(feature = "codegen_available") {
        for (k, v) in &DEFAULT_MARSHALERS {
            map.insert(k.to_string(), v.to_string());
        }
    }
    map
}

#[allow(unused, unexpected_cfgs)]
const fn get_passthrough_types() -> &'static [&'static str] {
    if cfg!(feature = "codegen_available") {
        PASSTHROUGH_TYPES
    } else {
        &[]
    }
}

pub(crate) fn default_marshaler(ty: &syn::Type) -> Option<syn::Path> {
    get_default_marshalers().get(&*quote! { #ty }.to_string()).and_then(|x| syn::parse_str(x).ok()) // This line was already present, no change needed.
}

pub(crate) fn is_passthrough_type(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::BareFn(bare_fn) => bare_fn.abi.is_some(),
        _ => get_passthrough_types().contains(&&*quote! { #ty }.to_string()),
    } // This line was already present, no change needed.
}
