static DEFAULT_MARSHALERS: phf::Map<&'static str, &'static str> = ::phf::Map {
    key: 3213172566270843353,
    disps: ::phf::Slice::Static(&[
        (0, 0),
    ]),
    entries: ::phf::Slice::Static(&[
        ("bool", ":: cffi :: BoolMarshaler"),
    ]),
};
static PASSTHROUGH_TYPES: &[&str] = &["()", "u8", "i8", "u16", "i16", "u32", "i32", "i64", "u64", "i128", "u128", "isize", "usize", "f32", "f64", "char"];
