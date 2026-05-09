using System;
using System.Collections.Generic;
using UnityEngine;

namespace TensorEngine.Core
{
    /// <summary>
    /// A linear (fully connected) layer: y = X @ W + b
    /// Mirrors Tensor-Engine's Linear module.
    /// </summary>
    [Serializable]
    public class Linear
    {
        [SerializeField] public int inFeatures;
        [SerializeField] public int outFeatures;
        [SerializeField] public bool hasBias;
        [SerializeField] Tensor weight;
        [SerializeField] Tensor bias;

        public Linear(int inFeatures, int outFeatures, bool bias = true)
        {
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
            this.hasBias = bias;
            // Kaiming uniform init
            float limit = Mathf.Sqrt(6f / (inFeatures + outFeatures));
            var wData = new float[inFeatures * outFeatures];
            var rng = new System.Random((int)System.DateTime.Now.Ticks);
            for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2 * limit - limit);
            weight = new Tensor(wData, new[] { inFeatures, outFeatures });
            if (hasBias) bias = Tensor.Zeros(new[] { outFeatures });
        }

        public Tensor Forward(Tensor input)
        {
            if (input.Rank < 2)
            {
                // 1D input: [d] -> [outFeatures]
                var r = new float[outFeatures];
                for (int j = 0; j < outFeatures; j++)
                {
                    float sum = 0f;
                    for (int i = 0; i < inFeatures; i++)
                        sum += input.data[i] * weight.data[i * outFeatures + j];
                    r[j] = sum;
                }
                if (hasBias && bias != null)
                    for (int j = 0; j < outFeatures; j++) r[j] += bias.data[j];
                return new Tensor(r, new[] { outFeatures });
            }
            // Batched: [batch, inFeatures] -> [batch, outFeatures]
            int batch = input.shape[0];
            var result = new float[batch * outFeatures];
            for (int b = 0; b < batch; b++)
            {
                int bOff = b * inFeatures;
                int rOff = b * outFeatures;
                for (int j = 0; j < outFeatures; j++)
                {
                    float sum = 0f;
                    for (int i = 0; i < inFeatures; i++)
                        sum += input.data[bOff + i] * weight.data[i * outFeatures + j];
                    result[rOff + j] = hasBias && bias != null ? sum + bias.data[j] : sum;
                }
            }
            return new Tensor(result, new[] { batch, outFeatures });
        }

        public Tensor GetWeight() => weight;
        public Tensor GetBias() => hasBias ? bias : null;
        public int InFeatures => inFeatures;
        public int OutFeatures => outFeatures;
        public bool Bias => hasBias;

        public override string ToString() => $"Linear({inFeatures}, {outFeatures}, bias={hasBias})";
    }

    /// <summary>
    /// Embedding layer: lookup table for token embeddings.
    /// </summary>
    [Serializable]
    public class Embedding
    {
        [SerializeField] public int vocabSize;
        [SerializeField] public int embedDim;
        [SerializeField] Tensor weight;

        public Embedding(int vocabSize, int embedDim)
        {
            this.vocabSize = vocabSize;
            this.embedDim = embedDim;
            float limit = Mathf.Sqrt(6f / (vocabSize + embedDim));
            var rng = new System.Random((int)System.DateTime.Now.Ticks);
            var wData = new float[vocabSize * embedDim];
            for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2 * limit - limit);
            weight = new Tensor(wData, new[] { vocabSize, embedDim });
        }

        public Tensor Forward(Tensor inputIds)
        {
            return TensorOps.EmbeddingLookup(weight, inputIds);
        }

        public Tensor Weight => weight;
        public int VocabSize => vocabSize;
        public int EmbedDim => embedDim;
    }

    /// <summary>
    /// Layer Normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
    /// </summary>
    [Serializable]
    public class LayerNorm
    {
        [SerializeField] public int normalizedDim;
        [SerializeField] public float eps;
        [SerializeField] Tensor gamma;
        [SerializeField] Tensor beta;

        public LayerNorm(int normalizedDim, float eps = 1e-5f)
        {
            this.normalizedDim = normalizedDim;
            this.eps = eps;
            gamma = Tensor.Zeros(new[] { normalizedDim });
            // Init gamma to ones for stable training
            for (int i = 0; i < normalizedDim; i++) gamma.data[i] = 1f;
            beta = Tensor.Zeros(new[] { normalizedDim });
        }

        public Tensor Forward(Tensor input)
        {
            // Normalize over the last dimension
            int[] shape = input.shape;
            int lastDim = shape[shape.Length - 1];
            int outer = 1;
            for (int i = 0; i < shape.Length - 1; i++) outer *= shape[i];
            int total = outer * lastDim;

            var result = new float[total];
            for (int i = 0; i < outer; i++)
            {
                float mean = 0f;
                for (int j = 0; j < lastDim; j++) mean += input.data[i * lastDim + j];
                mean /= lastDim;

                float var = 0f;
                for (int j = 0; j < lastDim; j++)
                {
                    float diff = input.data[i * lastDim + j] - mean;
                    var += diff * diff;
                }
                var /= lastDim;

                float invStd = 1f / Mathf.Sqrt(var + eps);
                for (int j = 0; j < lastDim; j++)
                {
                    result[i * lastDim + j] = gamma.data[j] * (input.data[i * lastDim + j] - mean) * invStd + beta.data[j];
                }
            }
            return new Tensor(result, shape);
        }

        public override string ToString() => $"LayerNorm({normalizedDim}, eps={eps})";
    }

    /// <summary>
    /// RMS Normalization (Llama-style): y = x / rms(x) * gamma
    /// No learnable beta. Used in Llama-style transformer blocks.
    /// </summary>
    [Serializable]
    public class RMSNorm
    {
        [SerializeField] public int normalizedDim;
        [SerializeField] float eps;
        [SerializeField] Tensor gamma;

        public RMSNorm(int normalizedDim, float eps = 1e-5f)
        {
            this.normalizedDim = normalizedDim;
            this.eps = eps;
            gamma = Tensor.Zeros(new[] { normalizedDim });
            for (int i = 0; i < normalizedDim; i++) gamma.data[i] = 1f;
        }

        public Tensor Forward(Tensor input)
        {
            int[] shape = input.shape;
            int lastDim = shape[shape.Length - 1];
            int outer = 1;
            for (int i = 0; i < shape.Length - 1; i++) outer *= shape[i];
            int total = outer * lastDim;

            var result = new float[total];
            for (int i = 0; i < outer; i++)
            {
                float rms = 0f;
                for (int j = 0; j < lastDim; j++)
                    rms += input.data[i * lastDim + j] * input.data[i * lastDim + j];
                rms = Mathf.Sqrt(rms / lastDim + eps);
                float scale = 1f / rms;
                for (int j = 0; j < lastDim; j++)
                    result[i * lastDim + j] = gamma.data[j] * input.data[i * lastDim + j] * scale;
            }
            return new Tensor(result, shape);
        }

        public override string ToString() => $"RMSNorm({normalizedDim})";
    }
}
