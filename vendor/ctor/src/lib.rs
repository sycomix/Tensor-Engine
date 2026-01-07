use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, Ident};

fn make_ident(prefix: &str, fn_ident: &Ident) -> Ident {
    let s = format!("__{}_{}", prefix, fn_ident);
    Ident::new(&s, fn_ident.span())
}

#[proc_macro_attribute]
pub fn ctor(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let vis = &input.vis;
    let sig = &input.sig;
    let attrs = &input.attrs;
    let block = &input.block;
    let fname = &input.sig.ident;

    let impl_ident = make_ident("ctor_impl", fname);
    let init_ident = make_ident("ctor_init", fname);
    let shim_ident = make_ident("ctor_shim", fname);

    let expanded = quote! {
        // keep original function name for normal calls, delegate to impl
        #(#attrs)*
        #vis #sig {
            #impl_ident();
        }

        // the real implementation function (kept private to avoid exporting items from proc-macro crates)
        #[allow(non_snake_case)]
        fn #impl_ident() {
            #block
        }

        // a private extern "C" shim that will be placed into the init array/section
        #[allow(non_snake_case)]
        extern "C" fn #shim_ident() {
            #impl_ident();
        }

        // place pointer in appropriate init section so platform will call it at startup
        #[used]
        #[cfg_attr(windows, link_section = ".CRT$XCU")]
        #[cfg_attr(not(windows), link_section = ".init_array")]
        static #init_ident: extern "C" fn() = #shim_ident;
    };

    // If the function we're wrapping is the common "init" used in cffi-impl, print its expansion for diagnosis
    if fname == "init" {
        eprintln!("[ctor debug] expanded for init: {}", expanded.to_string());
    }
    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn dtor(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // For simplicity mirror ctor behavior but put in dtor sections.
    let input = parse_macro_input!(item as ItemFn);
    let vis = &input.vis;
    let sig = &input.sig;
    let attrs = &input.attrs;
    let block = &input.block;
    let fname = &input.sig.ident;

    let impl_ident = make_ident("dtor_impl", fname);
    let init_ident = make_ident("dtor_fini", fname);
    let shim_ident = make_ident("dtor_shim", fname);

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            #impl_ident();
        }

        #[allow(non_snake_case)]
        fn #impl_ident() {
            #block
        }

        #[allow(non_snake_case)]
        extern "C" fn #shim_ident() {
            #impl_ident();
        }

        // .fini_array is a common dtor section on unix; windows has different patterns but
        // we include a generic marker here which may need platform-specific handling.
        #[used]
        #[cfg_attr(not(windows), link_section = ".fini_array")]
        #[cfg_attr(windows, link_section = ".CRT$XPU")]
        static #init_ident: extern "C" fn() = #shim_ident;
    };

    TokenStream::from(expanded)
}
