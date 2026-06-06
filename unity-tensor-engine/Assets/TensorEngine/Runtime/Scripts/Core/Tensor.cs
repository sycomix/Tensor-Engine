using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

namespace TensorEngine.Core
{
    /// <summary>
    /// Represents a multi-dimensional tensor, mirroring Tensor-Engine's Tensor structure.
    /// </summary>
    [Serializable]
    public class Tensor
    {
        [SerializeField] public float[] data;
        [SerializeField] public int[] shape;

        public int Rank => shape.Length;
        public int Length => data != null ? data.Length : 0;

        public Tensor() { }
        public Tensor(float[] data, int[] shape)
        {
            this.data = data;
            this.shape = shape;
        }
        public Tensor(List<float> data, int[] shape)
        {
            this.data = data.ToArray();
            this.shape = shape;
        }

        public Tensor(int[] tokens, int[] ints)
        {
            throw new NotImplementedException();
        }

        public static Tensor Scalar(float value) => new Tensor(new[] { value }, new int[0]);
        public static Tensor FromVector(float[] data) => new Tensor(data, new[] { data.Length });
        public static Tensor Zeros(int[] shape)
        {
            int len = shape.Aggregate(1, (a, b) => a * b);
            return new Tensor(new float[len], shape);
        }
        public static Tensor Identity(int size)
        {
            var d = new float[size * size];
            for (int i = 0; i < size; i++) d[i * size + i] = 1f;
            return new Tensor(d, new[] { size, size });
        }
        public static Tensor Random(int[] shape, System.Random rng = null)
        {
            int len = shape.Aggregate(1, (a, b) => a * b);
            var d = new float[len];
            var r = rng ?? new System.Random();
            for (int i = 0; i < len; i++) d[i] = (float)r.NextDouble();
            return new Tensor(d, shape);
        }

        public float this[int index] => data != null ? data[index] : 0f;
        public float this[int[] indices]
        {
            get
            {
                if (indices.Length != Rank)
                    throw new ArgumentException($"Expected {Rank} indices, got {indices.Length}");
                int flat = 0;
                for (int i = 0; i < Rank; i++) flat = flat * shape[i] + indices[i];
                return data != null ? data[flat] : 0f;
            }
        }

        public string ToJson() => JsonConvert.SerializeObject(new { data, shape });
        public static Tensor FromJson(string json)
        {
            var obj = JsonConvert.DeserializeAnonymousType(json, new { data = new float[0], shape = new int[0] });
            return new Tensor(obj.data, obj.shape);
        }

        public override string ToString() => $"Tensor(shape=[{string.Join(", ", shape)}], len={Length})";
    }

    /// <summary>
    /// Tensor operations mirroring Tensor-Engine's op layer.
    /// </summary>
    public static class TensorOps
    {
        public static Tensor Add(Tensor a, Tensor b)
        {
            if (a == null || b == null) throw new ArgumentNullException();
            if (a.Length != b.Length) throw new ArgumentException($"Shape mismatch: {a.shape} vs {b.shape}");
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = a.data[i] + b.data[i];
            return new Tensor(r, a.shape);
        }
        public static Tensor Sub(Tensor a, Tensor b)
        {
            if (a == null || b == null) throw new ArgumentNullException();
            if (a.Length != b.Length) throw new ArgumentException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = a.data[i] - b.data[i];
            return new Tensor(r, a.shape);
        }
        public static Tensor Mul(Tensor a, Tensor b)
        {
            if (a == null || b == null) throw new ArgumentNullException();
            if (a.Length != b.Length) throw new ArgumentException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = a.data[i] * b.data[i];
            return new Tensor(r, a.shape);
        }
        public static Tensor Div(Tensor a, Tensor b)
        {
            if (a == null || b == null) throw new ArgumentNullException();
            if (a.Length != b.Length) throw new ArgumentException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = a.data[i] / (b.data[i] + 1e-8f);
            return new Tensor(r, a.shape);
        }
        public static Tensor Neg(Tensor a)
        {
            if (a == null) throw new ArgumentNullException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = -a.data[i];
            return new Tensor(r, a.shape);
        }

        public static Tensor MatMul(Tensor a, Tensor b)
        {
            if (a == null || b == null) throw new ArgumentNullException();
            if (a.Rank < 2 || b.Rank < 2) throw new ArgumentException("MatMul requires at least 2D tensors");
            int m = a.shape[a.Rank - 2], k1 = a.shape[a.Rank - 1], k2 = b.shape[b.Rank - 2], n = b.shape[b.Rank - 1];
            if (k1 != k2) throw new ArgumentException($"Inner dims mismatch: {k1} vs {k2}");
            int[] batchShape = new int[a.Rank - 2];
            for (int i = 0; i < a.Rank - 2; i++) batchShape[i] = a.shape[i];
            int batchSize = batchShape.Aggregate(1, (a, b) => a * b);
            var result = new float[batchSize * m * n];
            for (int bIdx = 0; bIdx < batchSize; bIdx++)
            {
                int bOff = bIdx * m * k1, cOff = bIdx * m * n;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float sum = 0f;
                        for (int kk = 0; kk < k1; kk++) sum += a.data[bOff + i * k1 + kk] * b.data[bOff + kk * n + j];
                        result[cOff + i * n + j] = sum;
                    }
            }
            var resShape = new int[a.Rank];
            for (int i = 0; i < a.Rank - 2; i++) resShape[i] = a.shape[i];
            resShape[a.Rank - 2] = m; resShape[a.Rank - 1] = n;
            return new Tensor(result, resShape);
        }

        public static Tensor ReLU(Tensor a)
        {
            if (a == null) throw new ArgumentNullException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = Mathf.Max(0f, a.data[i]);
            return new Tensor(r, a.shape);
        }
        public static Tensor Sigmoid(Tensor a)
        {
            if (a == null) throw new ArgumentNullException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = 1f / (1f + Mathf.Exp(-a.data[i]));
            return new Tensor(r, a.shape);
        }
        public static Tensor Tanh(Tensor a)
        {
            if (a == null) throw new ArgumentNullException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = (float)Math.Tanh(a.data[i]);
            return new Tensor(r, a.shape);
        }
        public static Tensor Gelu(Tensor a)
        {
            if (a == null) throw new ArgumentNullException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                float x = a.data[i];
                r[i] = x * 0.5f * (1f + (float)Math.Tanh(Mathf.Sqrt(2f / Mathf.PI) * (x + 0.044715f * x * x * x)));
            }
            return new Tensor(r, a.shape);
        }
        public static Tensor Silu(Tensor a)
        {
            if (a == null) throw new ArgumentNullException();
            var s = TensorOps.Sigmoid(a);
            return TensorOps.Mul(a, s);
        }
        public static Tensor SwiGLU(Tensor a)
        {
            if (a == null) throw new ArgumentNullException();
            int half = a.shape[a.Rank - 1] / 2;
            var x = TensorOps.Slice(a, a.Rank - 1, 0, half);
            var gate = TensorOps.Slice(a, a.Rank - 1, half, half);
            var swig = new float[x.Length];
            for (int i = 0; i < x.Length; i++) swig[i] = x.data[i] * (1f / (1f + Mathf.Exp(-x.data[i])));
            var result = new float[a.Length];
            for (int i = 0; i < half; i++) result[i] = swig[i];
            for (int i = half; i < a.Length; i++) result[i] = x.data[i - half];
            return new Tensor(result, a.shape);
        }
        public static Tensor Softmax(Tensor a, int axis = -1)
        {
            if (a == null) throw new ArgumentNullException();
            if (a.Rank == 0) return a;
            if (axis < 0) axis = a.Rank - 1;
            var result = new float[a.Length];
            int axisSize = a.shape[axis], outerSize = 1, innerSize = 1;
            for (int i = 0; i < axis; i++) outerSize *= a.shape[i];
            for (int i = axis + 1; i < a.Rank; i++) innerSize *= a.shape[i];
            for (int i = 0; i < outerSize; i++)
            {
                float maxVal = float.MinValue;
                for (int j = 0; j < axisSize; j++)
                {
                    int idx = (i * axisSize + j) * innerSize;
                    if (a.data[idx] > maxVal) maxVal = a.data[idx];
                }
                float sumExp = 0f;
                for (int j = 0; j < axisSize; j++)
                {
                    int idx = (i * axisSize + j) * innerSize;
                    float expVal = Mathf.Exp(a.data[idx] - maxVal);
                    for (int k = 0; k < innerSize; k++) result[(i * axisSize + j) * innerSize + k] = expVal;
                    sumExp += expVal;
                }
                for (int j = 0; j < axisSize; j++)
                {
                    int idx = (i * axisSize + j) * innerSize;
                    for (int k = 0; k < innerSize; k++) result[idx + k] /= sumExp;
                }
            }
            return new Tensor(result, a.shape);
        }
        public static Tensor LogSoftmax(Tensor a, int axis = -1)
        {
            var sm = Softmax(a, axis);
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = Mathf.Log(sm.data[i] + 1e-8f);
            return new Tensor(r, a.shape);
        }
        public static Tensor Exp(Tensor a)
        {
            if (a == null) throw new ArgumentNullException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = Mathf.Exp(a.data[i]);
            return new Tensor(r, a.shape);
        }
        public static Tensor Log(Tensor a)
        {
            if (a == null) throw new ArgumentNullException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = Mathf.Log(a.data[i] + 1e-8f);
            return new Tensor(r, a.shape);
        }
        public static Tensor Pow(Tensor a, float exp)
        {
            if (a == null) throw new ArgumentNullException();
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = Mathf.Pow(a.data[i], exp);
            return new Tensor(r, a.shape);
        }
        public static Tensor Mean(Tensor a)
        {
            float sum = 0f;
            for (int i = 0; i < a.Length; i++) sum += a.data[i];
            return new Tensor(new[] { sum / a.Length }, new int[0]);
        }
        public static Tensor Sum(Tensor a)
        {
            float s = 0f;
            for (int i = 0; i < a.Length; i++) s += a.data[i];
            return new Tensor(new[] { s }, new int[0]);
        }
        public static Tensor Max(Tensor a)
        {
            float m = float.MinValue;
            for (int i = 0; i < a.Length; i++) if (a.data[i] > m) m = a.data[i];
            return new Tensor(new[] { m }, new int[0]);
        }
        public static Tensor Min(Tensor a)
        {
            float m = float.MaxValue;
            for (int i = 0; i < a.Length; i++) if (a.data[i] < m) m = a.data[i];
            return new Tensor(new[] { m }, new int[0]);
        }
        public static Tensor Reshape(Tensor a, int[] newShape)
        {
            int newLen = newShape.Aggregate(1, (a, b) => a * b);
            if (a.Length != newLen) throw new ArgumentException($"Cannot reshape {a.shape} to {newShape}");
            return new Tensor(a.data, newShape);
        }
        public static Tensor Transpose(Tensor a, int[] axes)
        {
            if (a.Rank != axes.Length) throw new ArgumentException();
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                int flatIdx = i, newIdx = 0;
                for (int dim = a.Rank - 1; dim >= 0; dim--)
                {
                    int size = a.shape[dim];
                    newIdx = axes[dim] * newIdx + (flatIdx % size);
                    flatIdx /= size;
                }
                result[newIdx] = a.data[i];
            }
            var newShape = new int[a.Rank];
            for (int i = 0; i < a.Rank; i++) newShape[i] = a.shape[axes[i]];
            return new Tensor(result, newShape);
        }
        public static Tensor Slice(Tensor a, int axis, int start, int length)
        {
            if (axis < 0 || axis >= a.Rank) throw new ArgumentException();
            int outer = 1, sliceSize = a.shape[axis], inner = 1;
            for (int i = 0; i < axis; i++) outer *= a.shape[i];
            for (int i = axis + 1; i < a.Rank; i++) inner *= a.shape[i];
            int outLen = outer * length * inner;
            var result = new float[outLen];
            for (int i = 0; i < outer; i++)
                for (int j = 0; j < length; j++)
                {
                    int srcIdx = (i * sliceSize + start + j) * inner;
                    int dstIdx = (i * length + j) * inner;
                    for (int k = 0; k < inner; k++) result[dstIdx + k] = a.data[srcIdx + k];
                }
            var newShape = new int[a.Rank];
            for (int i = 0; i < axis; i++) newShape[i] = a.shape[i];
            newShape[axis] = length;
            for (int i = axis + 1; i < a.Rank; i++) newShape[i] = a.shape[i];
            return new Tensor(result, newShape);
        }
        public static Tensor Concat(Tensor[] tensors, int axis)
        {
            if (tensors == null || tensors.Length == 0) throw new ArgumentException();
            int rank = tensors[0].Rank;
            if (axis < 0) axis += rank;
            for (int i = 1; i < tensors.Length; i++)
            {
                if (tensors[i].Rank != rank) throw new ArgumentException();
                for (int j = 0; j < rank; j++) if (j != axis && tensors[i].shape[j] != tensors[0].shape[j])
                    throw new ArgumentException();
            }
            int totalLen = 0;
            for (int i = 0; i < tensors.Length; i++)
            {
                int dimSize = tensors[i].shape[axis], others = 1;
                for (int j = 0; j < rank; j++) if (j != axis) others *= tensors[i].shape[j];
                totalLen += dimSize * others;
            }
            var result = new float[totalLen];
            int offset = 0;
            for (int i = 0; i < tensors.Length; i++)
            {
                int dimSize = tensors[i].shape[axis], others = 1;
                for (int j = 0; j < rank; j++) if (j != axis) others *= tensors[i].shape[j];
                Array.Copy(tensors[i].data, 0, result, offset, dimSize * others);
                offset += dimSize * others;
            }
            var newShape = new int[rank];
            for (int j = 0; j < rank; j++) newShape[j] = j == axis ? tensors.Sum(t => t.shape[j]) : tensors[0].shape[j];
            return new Tensor(result, newShape);
        }
        public static Tensor Stack(Tensor[] tensors, int axis)
        {
            if (tensors == null || tensors.Length == 0) throw new ArgumentException();
            var newShape = new int[tensors[0].Rank + 1];
            for (int i = 0; i < axis; i++) newShape[i] = tensors[0].shape[i];
            newShape[axis] = tensors.Length;
            for (int i = axis; i < tensors[0].Rank; i++) newShape[i + 1] = tensors[0].shape[i];
            int totalLen = newShape.Aggregate(1, (a, b) => a * b);
            var result = new float[totalLen];
            for (int t = 0; t < tensors.Length; t++)
            {
                for (int i = 0; i < tensors[t].Length; i++)
                {
                    int flatIdx = i, newIdx = 0, multiplier = 1;
                    for (int dim = newShape.Length - 1; dim >= 0; dim--)
                    {
                        if (dim == axis) newIdx += t * multiplier;
                        else
                        {
                            int dimIdx = flatIdx % tensors[t].shape[dim - (dim > axis ? 1 : 0)];
                            newIdx += dimIdx * multiplier;
                            flatIdx /= tensors[t].shape[dim - (dim > axis ? 1 : 0)];
                        }
                        multiplier *= newShape[dim];
                    }
                    result[newIdx] = tensors[t].data[i];
                }
            }
            return new Tensor(result, newShape);
        }
        public static Tensor Permute(Tensor a, int[] axes)
        {
            return TensorOps.Transpose(a, axes);
        }
        public static Tensor EmbeddingLookup(Tensor weights, Tensor indices)
        {
            if (weights == null || indices == null) throw new ArgumentNullException();
            if (weights.Rank < 2) throw new ArgumentException("Weights must be at least 2D");
            int vocabSize = weights.shape[0], dim = weights.shape[1];
            int[] idxShape = indices.shape;
            int seqLen = idxShape.Length;
            var result = new float[seqLen * dim];
            for (int i = 0; i < seqLen; i++)
            {
                int idx = (int)indices.data[i];
                for (int j = 0; j < dim; j++) result[i * dim + j] = weights.data[idx * dim + j];
            }
            var newShape = new int[idxShape.Length + 1];
            for (int i = 0; i < idxShape.Length; i++) newShape[i] = idxShape[i];
            newShape[newShape.Length - 1] = dim;
            return new Tensor(result, newShape);
        }
    }
}
