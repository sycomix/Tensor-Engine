/*
 * OpenCL stuff to run (some) of the tensor operations.
 */

use ocl::{
    enums::DeviceInfo, enums::DeviceInfoResult, Buffer, Context, Device, DeviceType, Event, Kernel,
    Platform, Program, Queue,
};
use std::alloc::Layout;
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[derive(Debug)]
#[allow(dead_code)]
struct Programs {
    matrix_mul_transposed_f16_program: Program,
    matrix_mul_transposed_f16: Kernel,
    matrix_mul_transposed_one_row_f16_program: Program,
    matrix_mul_transposed_one_row_f16: Kernel,
    matrix_mul_transposed_f16_cpu_optimized_program: Program,
    matrix_mul_transposed_f16_cpu_optimized: Kernel,
    silu_f16_program: Program,
    silu_f16: Kernel,
    hadamard_product_f16_program: Program,
    hadamard_product_f16: Kernel,
    transpose_f16_program: Program,
    transpose_f16: Kernel,
    rms_norm_f16_program: Program,
    rms_norm_f16: Kernel,
    rope_f16_program: Program,
    rope_f16: Kernel,
    softmax_f16_program: Program,
    softmax_f16: Kernel,
    attention_scores_f16_program: Program,
    attention_scores_f16: Kernel,
    attention_output_f16_program: Program,
    attention_output_f16: Kernel,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OpenCL {
    ctx: Context,
    queue: Queue,
    programs: Arc<RwLock<Programs>>,
    is_cpu_device: bool,
}

#[derive(Debug)]
pub struct OpenCLTensor {
    buf: Buffer<u16>, // really is f16
    initial_write_event: Option<ocl::Event>,
    last_event: Option<ocl::Event>,
    data: *const u16,
    data_layout: Layout,
    nitems: usize,
    rows: i64,
    cols: i64,
    cols_capacity: i64,
    queue: Queue,
    cl: OpenCL,
}

#[derive(Debug)]
pub struct OpenCLEvent {
    event: ocl::Event,
}

impl Drop for OpenCLTensor {
    fn drop(&mut self) {
        if self.initial_write_event.is_some() {
            self.initial_write_event
                .as_ref()
                .unwrap()
                .wait_for()
                .unwrap();
        }
        self.initial_write_event = None;
        if !self.data.is_null() {
            unsafe {
                std::alloc::dealloc(self.data as *mut u8, self.data_layout);
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum OpenCLError {
    #[error("OpenCL error: {0}")]
    OpenCL(#[from] ocl::Error),
    #[error("Cannot select device")]
    OpenCLDeviceSelection,
}

impl OpenCL {
    pub fn new(verbose: bool, nth_device: usize) -> Result<OpenCL, OpenCLError> {
        let platforms = Platform::list();
        let mut devices: Vec<(Platform, Device)> = Vec::new();
        for platform in platforms {
            for device in Device::list_all(platform)? {
                devices.push((platform, device));
            }
        }
        if verbose {
            println!("Enumerating OpenCL devices:");
        }
        for (idx, (_plat, device)) in devices.iter().enumerate() {
            if verbose {
                println!("OpenCL {} device: {}", idx, device.name()?,);
            }
        }
        if nth_device > devices.len() {
            return Err(OpenCLError::OpenCLDeviceSelection);
        }
        if verbose {
            println!("---");
            println!("Selected OpenCL device: {}", devices[nth_device].1.name()?);
        }

        let ctx = Context::builder()
            .platform(devices[nth_device].0)
            .devices(devices[nth_device].1)
            .build()?;

        let is_cpu_device = match devices[nth_device].1.info(DeviceInfo::Type)? {
            DeviceInfoResult::Type(DeviceType::CPU) => true,
            _ => false,
        };

        let queue = Queue::new(&ctx, devices[nth_device].1, None)?;
        let programs = make_programs(&ctx, &queue)?;
        Ok(OpenCL {
            ctx,
            queue,
            programs: Arc::new(RwLock::new(programs)),
            is_cpu_device,
        })
    }

    pub fn flush(&self) {
        let _ = self.queue.flush();
    }

    pub fn data_u16_to_gpu(
        &self,
        data: *const u16,
        data_layout: Layout,
        nitems: usize,
        rows: i64,
        cols: i64,
        cols_capacity: i64,
    ) -> Result<OpenCLTensor, OpenCLError> {
        unsafe {
            let buf = Buffer::builder()
                .queue(self.queue.clone())
                .len(nitems)
                .build()?;
            let mut event = Event::empty();
            let data_slice: &[u16] = std::slice::from_raw_parts(data, nitems);
            buf.cmd()
                .write(data_slice)
                .block(false)
                .enew(&mut event)
                .enq()?;
            Ok(OpenCLTensor {
                buf,
                initial_write_event: Some(event),
                last_event: None,
                data,
                data_layout,
                nitems,
                rows,
                cols,
                cols_capacity,
                queue: self.queue.clone(),
                cl: self.clone(),
            })
        }
    }
}

impl OpenCLTensor {
    pub fn cl(&self) -> OpenCL {
        self.cl.clone()
    }

    pub fn wait_until_ready(&mut self) {
        if self.last_event.is_some() {
            self.last_event.as_ref().unwrap().wait_for().unwrap();
            self.last_event = None;
        }
        if self.initial_write_event.is_some() {
            self.initial_write_event
                .as_ref()
                .unwrap()
                .wait_for()
                .unwrap();
            self.initial_write_event = None;
        }
        if !self.data.is_null() {
            unsafe {
                std::alloc::dealloc(self.data as *mut u8, self.data_layout);
            }
            self.data = std::ptr::null();
        }
    }

    pub fn data_u16_from_gpu(&mut self, data: *mut u16) -> Result<OpenCLEvent, OpenCLError> {
        unsafe {
            let mut event = Event::empty();
            let data_slice: &mut [u16] = std::slice::from_raw_parts_mut(data, self.nitems);
            let b = self
                .buf
                .cmd()
                .read(data_slice)
                .block(false)
                .enew(&mut event);
            b.enq()?;
            self.last_event = Some(event.clone());
            return Ok(OpenCLEvent { event });
        }
    }

    /// Copies all values from another tensor
    pub fn copy_inplace(&mut self, other: &OpenCLTensor) -> Result<OpenCLEvent, OpenCLError> {
        if other.rows != self.rows || other.cols != self.cols {
            panic!(
                "Cannot in-place copy tensors of different sizes: {}x{} <-- {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut event = Event::empty();
        other
            .buf
            .cmd()
            .queue(&other.queue)
            .copy(&self.buf, None, None)
            .enew(&mut event)
            .enq()?;
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }

    pub fn transpose_from(&mut self, other: &OpenCLTensor) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.write().unwrap();
        prg.transpose_f16.set_arg(0, self.buf.clone()).unwrap();
        prg.transpose_f16.set_arg(1, other.buf.clone()).unwrap();
        prg.transpose_f16
            .set_arg(2, self.cols_capacity as i32)
            .unwrap();
        prg.transpose_f16
            .set_arg(3, other.cols_capacity as i32)
            .unwrap();
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .transpose_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq().unwrap();
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }

    pub fn hadamard_product_inplace(
        &mut self,
        other: &OpenCLTensor,
    ) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.write().unwrap();
        prg.hadamard_product_f16.set_arg(0, self.buf.clone())?;
        prg.hadamard_product_f16.set_arg(1, other.buf.clone())?;
        prg.hadamard_product_f16
            .set_arg(2, self.cols_capacity as i32)?;
        prg.hadamard_product_f16
            .set_arg(3, other.cols_capacity as i32)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .hadamard_product_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }

    pub fn silu_inplace(&mut self) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.write().unwrap();
        prg.silu_f16.set_arg(0, self.buf.clone())?;
        prg.silu_f16.set_arg(1, self.cols_capacity as i32)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .silu_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }

    /// Apply per-head RMSNorm in-place. `weight` must have cols == head_dim.
    /// The data layout is [tokens, n_heads * head_dim] and the kernel processes
    /// each (token, head) pair independently.
    pub fn rms_norm_inplace(
        &mut self,
        weight: &OpenCLTensor,
        n_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.write().unwrap();
        prg.rms_norm_f16.set_arg(0, self.buf.clone())?;
        prg.rms_norm_f16.set_arg(1, weight.buf.clone())?;
        prg.rms_norm_f16.set_arg(2, self.cols_capacity as i32)?;
        prg.rms_norm_f16.set_arg(3, n_heads)?;
        prg.rms_norm_f16.set_arg(4, head_dim)?;
        prg.rms_norm_f16.set_arg(5, eps)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .rms_norm_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, n_heads as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }

    /// Apply RoPE in-place on both xq and xk tensors.
    /// xq: [tokens, n_q_heads * head_dim]
    /// xk: [tokens, n_kv_heads * head_dim]
    /// freqs_cos/sin: [max_seq_len, head_dim/2]  (f16)
    pub fn rope_inplace(
        &mut self,
        xk: &mut OpenCLTensor,
        freqs_cos: &OpenCLTensor,
        freqs_sin: &OpenCLTensor,
        n_q_heads: i32,
        head_dim: i32,
        group_size: i32,
        start_pos: i32,
    ) -> Result<OpenCLEvent, OpenCLError> {
        let half = head_dim / 2;
        let prg = self.cl.programs.write().unwrap();
        prg.rope_f16.set_arg(0, self.buf.clone())?;
        prg.rope_f16.set_arg(1, xk.buf.clone())?;
        prg.rope_f16.set_arg(2, freqs_cos.buf.clone())?;
        prg.rope_f16.set_arg(3, freqs_sin.buf.clone())?;
        prg.rope_f16.set_arg(4, self.cols_capacity as i32)?;
        prg.rope_f16.set_arg(5, xk.cols_capacity as i32)?;
        prg.rope_f16.set_arg(6, n_q_heads)?;
        prg.rope_f16.set_arg(7, head_dim)?;
        prg.rope_f16.set_arg(8, group_size)?;
        prg.rope_f16.set_arg(9, start_pos)?;
        let mut event = Event::empty();
        let n_tokens = self.rows;
        unsafe {
            let b = prg
                .rope_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([(n_tokens * n_q_heads as i64) as usize, half as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }

    /// Apply row-wise softmax in-place.
    pub fn softmax_inplace(&mut self) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.write().unwrap();
        prg.softmax_f16.set_arg(0, self.buf.clone())?;
        prg.softmax_f16.set_arg(1, self.cols_capacity as i32)?;
        prg.softmax_f16.set_arg(2, self.cols as i32)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .softmax_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, 1])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }

    /// Compute scores = Q * K^T for all query heads (GQA). Self is pre-allocated [n_q_heads, max_seq_len] f16 scores buffer.
    pub fn attention_scores_inplace(
        &mut self,
        q: &OpenCLTensor,
        k: &OpenCLTensor,
        n_q_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        kv_len: i32,
        scores_stride: i32,
        q_stride: i32,
        group_size: i32,
        scale: f32,
    ) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.write().unwrap();
        prg.attention_scores_f16.set_arg(0, q.buf.clone())?;
        prg.attention_scores_f16.set_arg(1, k.buf.clone())?;
        prg.attention_scores_f16.set_arg(2, self.buf.clone())?;
        prg.attention_scores_f16.set_arg(3, n_q_heads)?;
        prg.attention_scores_f16.set_arg(4, n_kv_heads)?;
        prg.attention_scores_f16.set_arg(5, head_dim)?;
        prg.attention_scores_f16.set_arg(6, kv_len)?;
        prg.attention_scores_f16.set_arg(7, scores_stride)?;
        prg.attention_scores_f16.set_arg(8, q_stride)?;
        prg.attention_scores_f16.set_arg(9, group_size)?;
        prg.attention_scores_f16.set_arg(10, scale)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .attention_scores_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([n_q_heads as usize, kv_len as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }

    /// Compute output = softmax(scores) * V for all query heads. Self is pre-allocated [n_q_heads, head_dim] f16 output buffer.
    pub fn attention_output_inplace(
        &mut self,
        scores: &OpenCLTensor,
        v: &OpenCLTensor,
        n_q_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        kv_len: i32,
        scores_stride: i32,
        out_stride: i32,
        v_stride: i32,
        group_size: i32,
    ) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.write().unwrap();
        prg.attention_output_f16.set_arg(0, scores.buf.clone())?;
        prg.attention_output_f16.set_arg(1, v.buf.clone())?;
        prg.attention_output_f16.set_arg(2, self.buf.clone())?;
        prg.attention_output_f16.set_arg(3, n_q_heads)?;
        prg.attention_output_f16.set_arg(4, n_kv_heads)?;
        prg.attention_output_f16.set_arg(5, head_dim)?;
        prg.attention_output_f16.set_arg(6, kv_len)?;
        prg.attention_output_f16.set_arg(7, scores_stride)?;
        prg.attention_output_f16.set_arg(8, out_stride)?;
        prg.attention_output_f16.set_arg(9, v_stride)?;
        prg.attention_output_f16.set_arg(10, group_size)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .attention_output_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([n_q_heads as usize, head_dim as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }

    pub fn matrix_mul_inplace_transposed(
        &mut self,
        src: &OpenCLTensor,
        other: &OpenCLTensor,
    ) -> Result<OpenCLEvent, OpenCLError> {
        if src.cols != other.cols {
            panic!(
                "OpenCL matrix_mul_inplace_transposed: src.cols must equal other.cols: {}x{} vs {}x{}",
                src.rows, src.cols, other.rows, other.cols
            );
        }
        if self.rows != src.rows || self.cols != other.rows {
            panic!(
                "OpenCL matrix_mul_inplace_transposed: self.rows must equal src.rows and self.cols must equal other.cols: {}x{} vs {}x{} vs {}x{}",
                self.rows, self.cols, src.rows, src.cols, other.rows, other.cols
            );
        }

        // Clear out the target memory.
        unsafe { self.buf.cmd().fill(0u16, None).block(false).enq()? };

        let prg = self.cl.programs.write().unwrap();

        // 0 = CPU optimized
        // 1 = GPU optimized
        // 2 = GPU optimized vector multiply (other.rows == 1)
        const CPU: u8 = 0;
        const GPU: u8 = 1;
        const GPU2: u8 = 2;
        let strategy: u8 = if self.cl.is_cpu_device {
            CPU
        } else {
            if src.rows == 1 {
                GPU2
            } else {
                GPU
            }
        };

        let prg = if strategy == CPU {
            &prg.matrix_mul_transposed_f16_cpu_optimized
        } else if strategy == GPU {
            &prg.matrix_mul_transposed_f16
        } else {
            &prg.matrix_mul_transposed_one_row_f16
        };
        prg.set_arg(0, self.buf.clone())?;
        prg.set_arg(1, src.buf.clone())?;
        prg.set_arg(2, other.buf.clone())?;
        prg.set_arg(3, src.cols_capacity as i32)?;
        prg.set_arg(4, other.cols_capacity as i32)?;
        prg.set_arg(5, self.cols_capacity as i32)?;
        prg.set_arg(6, self.rows as i32)?;
        prg.set_arg(7, self.cols as i32)?;
        prg.set_arg(8, src.cols as i32)?;
        let mut event = Event::empty();

        let rows16 = if self.rows % 16 == 0 {
            self.rows
        } else {
            self.rows + 16 - (self.rows % 16)
        };
        let cols16 = if self.cols % 16 == 0 {
            self.cols
        } else {
            self.cols + 16 - (self.cols % 16)
        };

        unsafe {
            if strategy == CPU {
                let b = prg
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size([self.cols as usize, self.rows as usize])
                    .enew(&mut event);
                b.enq()?;
            } else if strategy == GPU {
                let b = prg
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size([cols16 as usize, rows16 as usize])
                    .local_work_size([16, 16])
                    .enew(&mut event);
                b.enq()?;
            } else if strategy == GPU2 {
                let b = prg
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size([cols16 as usize, 1])
                    .local_work_size([16, 1])
                    .enew(&mut event);
                b.enq()?;
            } else {
                let b = prg
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size([self.cols as usize, self.rows as usize])
                    .enew(&mut event);
                b.enq()?;
            }
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }
}

impl OpenCLEvent {
    #[inline]
    pub fn wait(&self) {
        self.event.wait_for().unwrap();
    }
}

fn make_programs(ctx: &Context, queue: &Queue) -> Result<Programs, OpenCLError> {
    fn make_program_with_src(ctx: &Context, src: &str) -> Result<Program, OpenCLError> {
        let program = Program::builder().src(src).build(&ctx)?;
        Ok(program)
    }

    let matrix_mul_transposed_f16_program =
        make_program_with_src(ctx, MATRIX_MUL_TRANSPOSED_F16_SRC)?;
    let matrix_mul_transposed_f16 = Kernel::builder()
        .program(&matrix_mul_transposed_f16_program)
        .name("matrix_mul_transposed_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let matrix_mul_transposed_f16_cpu_optimized_program =
        make_program_with_src(ctx, MATRIX_MUL_TRANSPOSED_F16_CPU_OPTIMIZED_SRC)?;
    let matrix_mul_transposed_f16_cpu_optimized = Kernel::builder()
        .program(&matrix_mul_transposed_f16_cpu_optimized_program)
        .name("matrix_mul_transposed_f16_cpu_optimized")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let matrix_mul_transposed_one_row_f16_program =
        make_program_with_src(ctx, MATRIX_MUL_TRANSPOSED_F16_ONE_ROW_SRC)?;
    let matrix_mul_transposed_one_row_f16 = Kernel::builder()
        .program(&matrix_mul_transposed_one_row_f16_program)
        .name("matrix_mul_transposed_one_row_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let silu_f16_program = make_program_with_src(ctx, SILU_F16_SRC)?;
    let silu_f16 = Kernel::builder()
        .program(&silu_f16_program)
        .name("silu_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let hadamard_product_f16_program = make_program_with_src(ctx, HADAMARD_PRODUCT_F16_SRC)?;
    let hadamard_product_f16 = Kernel::builder()
        .program(&hadamard_product_f16_program)
        .name("hadamard_product_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let transpose_f16_program = make_program_with_src(ctx, TRANSPOSE_F16_SRC)?;
    let transpose_f16 = Kernel::builder()
        .program(&transpose_f16_program)
        .name("transpose_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let rms_norm_f16_program = make_program_with_src(ctx, RMS_NORM_F16_SRC)?;
    let rms_norm_f16 = Kernel::builder()
        .program(&rms_norm_f16_program)
        .name("rms_norm_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let rope_f16_program = make_program_with_src(ctx, ROPE_F16_SRC)?;
    let rope_f16 = Kernel::builder()
        .program(&rope_f16_program)
        .name("rope_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let softmax_f16_program = make_program_with_src(ctx, SOFTMAX_F16_SRC)?;
    let softmax_f16 = Kernel::builder()
        .program(&softmax_f16_program)
        .name("softmax_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let attention_scores_f16_program = make_program_with_src(ctx, ATTENTION_SCORES_F16_SRC)?;
    let attention_scores_f16 = Kernel::builder()
        .program(&attention_scores_f16_program)
        .name("attention_scores_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let attention_output_f16_program = make_program_with_src(ctx, ATTENTION_OUTPUT_F16_SRC)?;
    let attention_output_f16 = Kernel::builder()
        .program(&attention_output_f16_program)
        .name("attention_output_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    Ok(Programs {
        matrix_mul_transposed_f16_program,
        matrix_mul_transposed_f16,
        matrix_mul_transposed_one_row_f16_program,
        matrix_mul_transposed_one_row_f16,
        matrix_mul_transposed_f16_cpu_optimized_program,
        matrix_mul_transposed_f16_cpu_optimized,
        silu_f16_program,
        silu_f16,
        hadamard_product_f16_program,
        hadamard_product_f16,
        transpose_f16_program,
        transpose_f16,
        rms_norm_f16_program,
        rms_norm_f16,
        rope_f16_program,
        rope_f16,
        softmax_f16_program,
        softmax_f16,
        attention_scores_f16_program,
        attention_scores_f16,
        attention_output_f16_program,
        attention_output_f16,
    })
}

const MATRIX_MUL_TRANSPOSED_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matrix_mul_transposed_f16(
    __global half *tgt,
    __global const half *left,
    __global const half *right,
    const int left_cols_capacity,
    const int right_cols_capacity,
    const int ncols_capacity,
    const int nrows,
    const int ncols,  // size of target
    const int shared_sz
) {
    __local float lefttile[16][16];
    __local float righttile[16][16];

    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int num_tiles = (shared_sz + 15) / 16;

    float sum = 0.0f;
    for (int t = 0; t < num_tiles; ++t) {
        if (global_y < nrows) {
            lefttile[local_y][local_x] = vload_half(global_y * left_cols_capacity + t * 16 + local_x, left);
        } else {
            lefttile[local_y][local_x] = 0.0f;
        }
        if (global_x < ncols) {
            righttile[local_y][local_x] = vload_half(global_x * right_cols_capacity + t * 16 + local_y, right);
        } else {
            righttile[local_y][local_x] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; ++k) {
            sum += lefttile[local_y][k] * righttile[k][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_x < ncols && global_y < nrows) {
        vstore_half(sum, global_y * ncols_capacity + global_x, (__global half*) tgt);
    }
}
"#;

const MATRIX_MUL_TRANSPOSED_F16_ONE_ROW_SRC: &str = r#"
__kernel void matrix_mul_transposed_one_row_f16(
    __global half *tgt,
    __global const half *left,
    __global const half *right,
    const int left_cols_capacity,
    const int right_cols_capacity,
    const int ncols_capacity,
    const int nrows,
    const int ncols,  // size of target
    const int shared_sz
) {
    // assertions:
    // nrows == 1
    // left_rows == 1
    __local float lefttile[16];
    __local float righttile[16][16];

    const int global_x = get_global_id(0);
    const int local_x = get_local_id(0);
    const int num_tiles = (shared_sz + 15) / 16;
    const int x_tile = (global_x / 16) * 16;

    float sum = 0.0f;
    if (x_tile + 15 < ncols) {
        for (int t = 0; t < num_tiles; ++t) {
            lefttile[local_x] = vload_half(t * 16 + local_x, left);
            for (int k = 0; k < 16; ++k) {
                righttile[k][local_x] = vload_half(t * 16 + local_x + (x_tile + k) * right_cols_capacity, right);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < 16; ++k) {
                sum += lefttile[k] * righttile[local_x][k];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    } else {
        for (int t = 0; t < num_tiles; ++t) {
            lefttile[local_x] = vload_half(t * 16 + local_x, left);
            for (int k = 0; k < 16; ++k) {
                if (x_tile + k >= ncols) {
                    righttile[k][local_x] = 0.0f;
                } else {
                    righttile[k][local_x] = vload_half(t * 16 + local_x + (x_tile + k) * right_cols_capacity, right);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < 16; ++k) {
                sum += lefttile[k] * righttile[local_x][k];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (global_x < ncols) {
        vstore_half(sum, global_x, (__global half*) tgt);
    }
}"#;

const MATRIX_MUL_TRANSPOSED_F16_CPU_OPTIMIZED_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matrix_mul_transposed_f16_cpu_optimized(
    __global half *tgt,
    __global const half *left,
    __global const half *right,
    const int left_cols_capacity,
    const int right_cols_capacity,
    const int ncols_capacity,
    const int nrows,
    const int ncols,  // size of target
    const int shared_sz
) {
    const int tgt_col = get_global_id(0);
    const int tgt_row = get_global_id(1);
    int col_iterations = shared_sz / 16;
    if (shared_sz % 16 != 0) {
        col_iterations = col_iterations + 1;
    }
    float16 sum = 0;
    for (int col16 = 0; col16 < col_iterations; col16++) {
        const float16 left8 = vload_half16((tgt_row * left_cols_capacity)/16 + col16, (__global const half*) left);
        const float16 right8 = vload_half16((tgt_col * right_cols_capacity)/16 + col16, (__global const half*) right);
        // hadamard product FMA add it to sum
        // const float16 result8 = left8 * right8;
        // sum += result8;
        sum = fma(left8, right8, sum);
    }
    // Reduce as accurately as possible
    float sum1 = sum.s0 + sum.s1;
    float sum2 = sum.s2 + sum.s3;
    float sum3 = sum.s4 + sum.s5;
    float sum4 = sum.s6 + sum.s7;
    float sum5 = sum.s8 + sum.s9;
    float sum6 = sum.sa + sum.sb;
    float sum7 = sum.sc + sum.sd;
    float sum8 = sum.se + sum.sf;
    float sum11 = sum1 + sum2;
    float sum12 = sum3 + sum4;
    float sum13 = sum5 + sum6;
    float sum14 = sum7 + sum8;
    float sum21 = sum11 + sum12;
    float sum22 = sum13 + sum14;
    float total = sum21 + sum22;
    vstore_half(total, 0, (__global half*) &tgt[tgt_row * ncols_capacity + tgt_col]);
}
"#;

/// Computes SILU for every f16 value in the tensor
const SILU_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void silu_f16(__global half *tgt,
                       const int ncols_capacity)
{
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const float val = vload_half(tgt_row * ncols_capacity + tgt_col, (__global const half*) tgt);
    const float result = val * (1.0 / (1.0 + exp(-val)));
    vstore_half(result, tgt_row * ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;

/// Computes hadamard product of two identially sized tensors
const HADAMARD_PRODUCT_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void hadamard_product_f16(__global half *tgt,
                                   __global const half *left,
                                   const int ncols_capacity,
                                   const int left_cols_capacity) {
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const float tgt_value = vload_half(tgt_row * ncols_capacity + tgt_col, (__global const half*) tgt);
    const float left_value = vload_half(tgt_row * left_cols_capacity + tgt_col, (__global const half*) left);
    const float result = tgt_value * left_value;
    vstore_half(result, tgt_row * ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;

/// Computes the transpose of a matrix
const TRANSPOSE_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void transpose_f16(__global half *tgt,
                            __global const half *left,
                            const int ncols_capacity,
                            const int left_cols_capacity)
{
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const int src_row = tgt_col;
    const int src_col = tgt_row;
    const float val = vload_half(src_row * left_cols_capacity + src_col, (__global const half*) left);
    vstore_half(val, tgt_row * ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;

/// Computes per-head RMSNorm on a [tokens, n_heads * head_dim] tensor in-place.
const RMS_NORM_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rms_norm_f16(
    __global half *data,
    __global const half *weight,
    const int ncols_capacity,
    const int n_heads,
    const int head_dim,
    const float eps
) {
    const int token = get_global_id(0);
    const int head = get_global_id(1);
    const int n_tokens = get_global_size(0);
    if (token >= n_tokens) return;

    int row_offset = token * ncols_capacity + head * head_dim;

    float sum_sq = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float val = vload_half(row_offset + i, (__global const half*) data);
        sum_sq += val * val;
    }

    float rms = rsqrt(sum_sq / head_dim + eps);

    for (int i = 0; i < head_dim; i++) {
        float val = vload_half(row_offset + i, (__global const half*) data);
        float w = vload_half(i, weight);
        vstore_half(val * rms * w, row_offset + i, (__global half*) data);
    }
}
"#;

/// Applies Rotary Position Embedding to Q and K tensors in-place.
/// xq: [tokens, n_q_heads * head_dim]
/// xk: [tokens, n_kv_heads * head_dim]
/// freqs_cos/sin: [max_seq_len, head_dim/2]  (cos/sin values as f16)
const ROPE_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rope_f16(
    __global half *xq,
    __global half *xk,
    __global const half *freqs_cos,
    __global const half *freqs_sin,
    const int xq_cols_capacity,
    const int xk_cols_capacity,
    const int n_q_heads,
    const int head_dim,
    const int group_size,
    const int start_pos
) {
    const int token_head = get_global_id(0);
    const int col = get_global_id(1);
    const int total_work = get_global_size(0);
    const int half_dim = get_global_size(1);
    if (col >= half_dim) return;
    if (token_head >= total_work) return;

    int token = token_head / n_q_heads;
    int head_q = token_head % n_q_heads;
    int head_kv = head_q / group_size;
    int pos = start_pos + token;

    float cos_val = vload_half(pos * half_dim + col, freqs_cos);
    float sin_val = vload_half(pos * half_dim + col, freqs_sin);

    int xq_offset = token * xq_cols_capacity + head_q * head_dim;
    float q_real = vload_half(xq_offset + col, (__global const half*) xq);
    float q_imag = vload_half(xq_offset + col + half_dim, (__global const half*) xq);
    vstore_half(q_real * cos_val - q_imag * sin_val, xq_offset + col, (__global half*) xq);
    vstore_half(q_real * sin_val + q_imag * cos_val, xq_offset + col + half_dim, (__global half*) xq);

    int xk_offset = token * xk_cols_capacity + head_kv * head_dim;
    float k_real = vload_half(xk_offset + col, (__global const half*) xk);
    float k_imag = vload_half(xk_offset + col + half_dim, (__global const half*) xk);
    vstore_half(k_real * cos_val - k_imag * sin_val, xk_offset + col, (__global half*) xk);
    vstore_half(k_real * sin_val + k_imag * cos_val, xk_offset + col + half_dim, (__global half*) xk);
}
"#;

/// Computes row-wise softmax on a [rows, cols] tensor in-place.
const SOFTMAX_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void softmax_f16(
    __global half *data,
    const int ncols_capacity,
    const int ncols
) {
    const int row = get_global_id(0);
    const int nrows = get_global_size(0);
    if (row >= nrows) return;

    float max_val = -INFINITY;
    for (int i = 0; i < ncols; i++) {
        float val = vload_half(row * ncols_capacity + i, (__global const half*) data);
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (int i = 0; i < ncols; i++) {
        float val = vload_half(row * ncols_capacity + i, (__global const half*) data);
        sum += exp(val - max_val);
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < ncols; i++) {
        float val = vload_half(row * ncols_capacity + i, (__global const half*) data);
        vstore_half(exp(val - max_val) * inv_sum, row * ncols_capacity + i, (__global half*) data);
    }
}
"#;

/// Computes Q*K^T for all query heads with GQA support.
/// Q: [n_q_heads, head_dim] with stride q_stride between heads.
/// K: [n_kv_heads * max_seq_len, head_dim] (each KV head has positions×dim contiguous, stride=head_dim).
/// scores: [n_q_heads, max_seq_len] with stride scores_stride between heads (only kv_len positions valid).
const ATTENTION_SCORES_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void attention_scores_f16(
    __global const half *q,
    __global const half *k,
    __global half *scores,
    const int n_q_heads,
    const int n_kv_heads,
    const int head_dim,
    const int kv_len,
    const int scores_stride,
    const int q_stride,
    const int group_size,
    const float scale
) {
    const int head = get_global_id(0);
    const int pos = get_global_id(1);
    if (head >= n_q_heads || pos >= kv_len) return;
    const int kv_head = head / group_size;

    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        sum += vload_half(head * q_stride + d, q) * vload_half((kv_head * kv_len + pos) * head_dim + d, k);
    }
    vstore_half(sum * scale, head * scores_stride + pos, scores);
}
"#;

/// Computes softmax(scores) * V for all query heads.
/// scores: [n_q_heads, max_seq_len] with stride scores_stride between heads (only kv_len valid).
/// V: [n_kv_heads * head_dim, max_seq_len] with stride v_stride between rows of same dim.
/// output: [n_q_heads, head_dim] with stride out_stride between heads.
const ATTENTION_OUTPUT_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void attention_output_f16(
    __global const half *scores,
    __global const half *v,
    __global half *output,
    const int n_q_heads,
    const int n_kv_heads,
    const int head_dim,
    const int kv_len,
    const int scores_stride,
    const int out_stride,
    const int v_stride,
    const int group_size
) {
    const int head = get_global_id(0);
    const int d = get_global_id(1);
    if (head >= n_q_heads || d >= head_dim) return;
    const int kv_head = head / group_size;

    float sum = 0.0f;
    for (int pos = 0; pos < kv_len; pos++) {
        sum += vload_half(head * scores_stride + pos, scores) * vload_half((kv_head * head_dim + d) * v_stride + pos, v);
    }
    vstore_half(sum, head * out_stride + d, output);
}
"#;
