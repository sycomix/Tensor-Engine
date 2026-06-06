#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub mod driver {
include!(concat!(env!("OUT_DIR"), "/driver_bind.rs"));
}

pub mod driver_types {
    use super::driver::{CUevent_st, CUstream_st, CUuuid_st};
    include!(concat!(env!("OUT_DIR"), "/driver_types_bind.rs"));
}

pub mod library_types {
    pub type size_t = usize;
    include!(concat!(env!("OUT_DIR"), "/libtypes_bind.rs"));
}

pub mod runtime {
    use super::driver_types::{
        cudaDeviceAttr, cudaError_t, cudaEvent_t, cudaGraphicsResource_t,
        cudaMemLocation, cudaMemoryAdvise, cudaMemRangeAttribute,
        cudaMemcpyKind, cudaStream_t, cudaUUID_t,
    };
    use super::library_types::size_t;

    include!(concat!(env!("OUT_DIR"), "/runtime_bind.rs"));
}
