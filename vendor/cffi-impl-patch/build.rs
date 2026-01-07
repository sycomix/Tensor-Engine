use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use quote::quote;
use syn::Type;

macro_rules! map_marshalers { [$($rust:ty => $c:ty,)*] => {{
    let mut map = std::collections::HashMap::<String, syn::Path>::new();
    $(map.insert(
        quote!{ $rust }.to_string(),
        syn::parse2(quote!{ $c }).expect(concat!("cannot parse", stringify!($c), "as path")),
    );)*
    map
}}}

macro_rules! type_array { [$($rust:ty,)*] => {{
    vec![ $(syn::parse2(quote!{ $rust }).expect(concat!("cannot parse", stringify!($rust), "as type")), )* ]
}}}

fn main() {
    let default_marshalers: HashMap<String, syn::Path> = map_marshalers![
        bool => ::cffi::BoolMarshaler,
    ];

    let path = Path::new(&env::var("OUT_DIR").unwrap()).join("codegen.rs");
    let mut file = BufWriter::new(File::create(&path).unwrap());

    write!(&mut file, "static DEFAULT_MARSHALERS: phf::Map<&'static str, &'static str> = ").unwrap();
    let mut map = phf_codegen::Map::new();
    for (key, value) in default_marshalers.iter() {
        map.entry(key.clone(), &format!("\"{}\"", quote! { #value }.to_string()));
    }
    // phf_codegen v0.7 writes into the provided writer
    map.build(&mut file).unwrap();
    write!(&mut file, ";\n").unwrap();

    let types: Vec<Type> = type_array![
        (), u8, i8, u16, i16, u32, i32, i64, u64, i128, u128, isize, usize, f32, f64, char,
    ];

    write!(&mut file, "static PASSTHROUGH_TYPES: &[&str] = &[\"{}\"];\n", types.into_iter().map(|x| quote! { #x }.to_string()).collect::<Vec<_>>().join("\", \"")).unwrap();
}