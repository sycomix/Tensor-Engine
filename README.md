# **Tensor Engine: Documentation**

**Tensor Engine** is a lightweight, machine learning framework built in Rust. It features a define-by-run automatic differentiation engine (Autograd), a suite of neural network primitives, and efficient Python bindings via PyO3.

## **1\. Architecture Overview**

The library is built on three core pillars:

1. **The Tensor**: A thread-safe wrapper around ndarray that tracks computational history.  
2. **The Operation Trait**: A unified interface for defining forward and backward passes.  
3. **The Module System**: High-level abstractions for layers, optimizers, and models.

### **Data Flow**

1. **Storage**: Data is stored in ndarray::ArrayD\<f32\> (dynamic dimensional arrays).  
2. **Memory Management**: Tensors use Arc\<Mutex\<TensorData\>\> to share data and gradients safely across threads and computation graphs.  
3. **Compute**: Operations (like MatMul, Relu) consume input Tensors and produce output Tensors, recording the "lineage" (the operation and input parents) to build a Directed Acyclic Graph (DAG) for backpropagation.

## **2\. Developer Guide (Rust Internals)**

### **2.1 The Tensor Structure**

Located in src/tensor.rs, the Tensor struct is the heart of the engine. It is a smart pointer wrapper ensuring cheap cloning (shallow copies).

// Simplified view of the internal structure  
pub struct TensorData {  
    pub data: ArrayD\<f32\>,                 // Raw data  
    pub grad: Option\<ArrayD\<f32\>\>,         // Accumulated gradient  
    pub creator: Option\<Arc\<dyn Operation\>\>, // The op that created this tensor  
    pub inputs: Vec\<Tensor\>,               // Parents in the DAG  
    pub requires\_grad: bool,               // Flag to track gradients  
}

**Key Behaviors:**

* **Broadcasting**: Element-wise operations (Add, Sub, Mul, Div) automatically broadcast shapes (e.g., adding \[3, 1\] to \[3, 4\]).  
* **Locking**: Accessing data requires calling .lock(), which returns a MutexGuard. This prevents data races during parallel operations.

### **2.2 The Autograd System**

Backpropagation is implemented via the backward() method on the Tensor struct.

1. **Topological Descent**: It starts at the loss node (root).  
2. **Gradient Propagation**: It calls the backward method of the creator Operation.  
3. **Accumulation**: Gradients returned by the operation are accumulated (+=) into the .grad field of the input tensors.  
4. **Recursion**: The process repeats recursively for input tensors that require gradients.

### **2.3 The Operation Trait**

Located in src/ops.rs, every differentiable function implements this trait:

pub trait Operation: Send \+ Sync {  
    fn forward(\&self, inputs: &\[Tensor\], output: \&mut ArrayD\<f32\>);  
    fn backward(\&self, inputs: &\[Tensor\], output\_grad: \&ArrayD\<f32\>) \-\> Vec\<ArrayD\<f32\>\>;  
}

**Supported Operations:**

* **Arithmetic**: Add, Sub, Mul, Div, Pow  
* **Linear Algebra**: MatMul (Native Rust or OpenBLAS backed)  
* **Activation**: ReLU, Sigmoid, Tanh, Softmax, LogSoftmax  
* **CNN Ops**: Conv2D (NCHW layout), MaxPool2D  
* **Shapes**: Reshape, Transpose, Concat, Stack, Slice, Flatten  
* **Loss Ops**: MSELoss, CrossEntropyLogits (fused/stable), NLLLoss  
* **Normalization**: LayerNorm (with learnable affine parameters)

### **2.4 Neural Network Modules**

Located in src/nn/mod.rs, the Module trait defines stateful layers.

* **Linear**: Fully connected layer with optional bias.  
* **Conv2D**: Wraps the Conv2D op, managing weight/bias tensors.  
* **RNN/LSTM**: Unrolled implementations of recurrent cells.  
* **TransformerBlock**: Implements self-attention and feed-forward blocks with residuals.  
* **Optimizers**: SGD (with momentum) and Adam.

## **3\. Python Bindings API**

The Python API mirrors the Rust API but is exposed via src/lib.rs using pyo3.

### **Setup**

pip install maturin  
maturin develop \--release

### **Core API Reference**

#### **Tensor**

The main class for data manipulation.

* **Creation**: te.Tensor(data\_list, shape\_list)  
* **Properties**: shape, requires\_grad, grad  
* **Methods**:  
  * backward(): Triggers autograd.  
  * numpy() / get\_data(): Exports data.  
  * reshape(shape), transpose()  
  * **Ops**: \+, \-, \*, /, \*\* (pow), matmul(@) implied.  
  * **Activations**: relu(), sigmoid(), tanh(), softmax(axis), log\_softmax(axis).

#### **NN Layers**

* te.Linear(in\_features, out\_features, bias=True)  
* te.CrossEntropyLoss(): Standard classification loss.  
* te.SoftmaxCrossEntropyLoss(): Fused efficient implementation.  
* te.MSELoss(): Mean Squared Error.

#### **Optimizers**

* te.SGD(lr, momentum)  
* te.Adam(lr, beta1, beta2, eps)

## **4\. How to Extend the Library**

### **Adding a New Operation (Rust)**

To add a new mathematical operation (e.g., LeakyReLU), follow these steps:

1. **Define the Struct** (src/ops.rs):  
   pub struct LeakyReLU { pub alpha: f32 }

2. **Implement Operation Trait**:  
   impl Operation for LeakyReLU {  
       fn forward(\&self, inputs: &\[Tensor\], output: \&mut ArrayD\<f32\>) {  
           let x \= \&inputs\[0\].lock().data;  
           \*output \= x.mapv(|v| if v \> 0.0 { v } else { self.alpha \* v });  
       }

       fn backward(\&self, inputs: &\[Tensor\], output\_grad: \&ArrayD\<f32\>) \-\> Vec\<ArrayD\<f32\>\>;  
           let x \= \&inputs\[0\].lock().data;  
           // Gradient is 1.0 if x \> 0 else alpha  
           let grad\_input \= output\_grad \* x.mapv(|v| if v \> 0.0 { 1.0 } else { self.alpha });  
           vec\!\[grad\_input\]  
       }

       fn as\_any(\&self) \-\> \&dyn Any { self }  
   }

3. **Expose Method on Tensor** (src/tensor.rs):  
   pub fn leaky\_relu(\&self, alpha: f32) \-\> Tensor {  
       Tensor::apply(Arc::new(LeakyReLU { alpha }), &\[self.clone()\])  
   }

### **Adding a New Layer (Rust)**

To add a high-level layer (e.g., GRU):

1. Define a struct in src/nn/mod.rs holding the learnable weights (Tensor).  
2. Implement the Module trait.  
3. Implement forward using existing primitives (matmul, sigmoid, etc.).  
4. Return weights in parameters().

## **5\. Performance Considerations**

* **BLAS Support**: The library supports OpenBLAS for matrix multiplication. Use the feature flag \--features openblas to enable it. This significantly speeds up MatMul and Conv2D operations compared to the default pure-Rust implementation.  
* **Memory Overhead**: The Arc\<Mutex\<\>\> wrapper adds minor overhead per tensor. For very small tensors (scalars), this overhead is noticeable. It is optimized for batched operations (matrix-matrix ops).  
* **Clone vs Ref**: Tensors are cheap to clone (pointer copy). However, accessing data (.lock()) is costly if done frequently in a tight loop. Prefer vectorized operations (e.g., x.add(y)) over iterating through elements manually.

## **6\. Testing**

* **Unit Tests**: Run cargo test to execute Rust unit tests in src/nn/tests and tests/.  
* **Python Smoke Test**: Run python tests/python\_smoke\_test.py to verify the Python extension functionality.  
* **Benchmarks**: Benchmarks are located in benches/. Run cargo bench to check performance regressions, particularly for MatMul.