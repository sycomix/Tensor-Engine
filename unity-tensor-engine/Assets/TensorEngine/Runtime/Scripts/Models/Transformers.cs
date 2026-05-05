using System;
using System.Collections.Generic;
using UnityEngine;

namespace TensorEngine.Models
{
    /// <summary>
    /// Rotary Positional Embeddings (RoPE) for transformer attention.
    /// Mirrors Tensor-Engine's RoPE implementation.
    /// </summary>
    public static class Rope
    {
        public static (Tensor q, Tensor k) Apply(Tensor q, Tensor k, int headDim, int seqLen, float theta = 10000f)
        {
            // q, k shape: [batch, seqLen, headDim]
            // Apply RoPE per sequence position
            int batch = q.shape[0];
            var qOut = new float[q.Length];
            var kOut = new float[k.Length];

            for (int b = 0; b < batch; b++)
            {
                for (int pos = 0; pos < seqLen; pos++)
                {
                    float freq = 1f / Mathf.Pow(theta, (2f * (int)(headDim / 2)) / (float)headDim);
                    for (int i = 0; i < headDim / 2; i++)
                    {
                        float angle = pos * freq;
                        float cos = Mathf.Cos(angle);
                        float sin = Mathf.Sin(angle);
                        int qOff = b * seqLen * headDim + pos * headDim + i * 2;
                        int kOff = b * seqLen * headDim + pos * headDim + i * 2;

                        // Apply to Q
                        float q0 = q.data[qOff], q1 = q.data[qOff + 1];
                        qOut[qOff] = q0 * cos - q1 * sin;
                        qOut[qOff + 1] = q0 * sin + q1 * cos;

                        // Apply to K
                        float k0 = k.data[kOff], k1 = k.data[kOff + 1];
                        kOut[kOff] = k0 * cos - k1 * sin;
                        kOut[kOff + 1] = k0 * sin + k1 * cos;
                    }
                }
            }
            return (new Tensor(qOut, q.shape), new Tensor(kOut, k.shape));
        }
    }

    /// <summary>
    /// Multi-Head Attention layer.
    /// Mirrors Tensor-Engine's MultiHeadAttention.
    /// </summary>
    [Serializable]
    public class MultiHeadAttention
    {
        public Linear qProj, kProj, vProj, oProj;
        public int numHeads;
        public int headDim;
        public int dModel;
        public bool useRope;
        public float ropeTheta;

        public MultiHeadAttention(int dModel, int numHeads, bool useRope = false, float ropeTheta = 10000f)
        {
            this.dModel = dModel;
            this.numHeads = numHeads;
            this.headDim = dModel / numHeads;
            this.useRope = useRope;
            this.ropeTheta = ropeTheta;
            qProj = new Linear(dModel, dModel);
            kProj = new Linear(dModel, dModel);
            vProj = new Linear(dModel, dModel);
            oProj = new Linear(dModel, dModel);
        }

        public Tensor Forward(Tensor input, bool causal = false, Tensor mask = null)
        {
            int batch = input.shape[0];
            int seqLen = input.shape[1];

            // Q, K, V projections
            Tensor q = qProj.Forward(input);
            Tensor k = kProj.Forward(input);
            Tensor v = vProj.Forward(input);

            // Reshape to [batch, seqLen, numHeads, headDim]
            q = TensorOps.Reshape(q, new[] { batch, seqLen, numHeads, headDim });
            k = TensorOps.Reshape(k, new[] { batch, seqLen, numHeads, headDim });
            v = TensorOps.Reshape(v, new[] { batch, seqLen, numHeads, headDim });

            // Permute to [batch, numHeads, seqLen, headDim]
            q = TensorOps.Permute(q, new[] { 0, 2, 1, 3 });
            k = TensorOps.Permute(k, new[] { 0, 2, 1, 3 });
            v = TensorOps.Permute(v, new[] { 0, 2, 1, 3 });

            // Apply RoPE
            if (useRope)
            {
                var (qRope, kRope) = Rope.Apply(q, k, headDim, seqLen, ropeTheta);
                q = qRope;
                k = kRope;
            }

            // Scaled dot-product attention
            // q: [batch, heads, seqLen, headDim], k: [batch, heads, headDim, seqLen]
            var kT = TensorOps.Permute(k, new[] { 0, 1, 3, 2 });
            float scale = 1f / Mathf.Sqrt(headDim);

            // Batch matmul: [batch, heads, seqLen, seqLen]
            Tensor attnScores = TensorOps.MatMul(q, kT);
            // Scale
            var scaleTensor = Tensor.Scalar(scale);
            // Element-wise multiply (broadcast)
            var attnScaled = new float[attnScores.Length];
            for (int i = 0; i < attnScores.Length; i++) attnScaled[i] = attnScores.data[i] * scale;
            attnScores = new Tensor(attnScaled, attnScores.shape);

            // Causal mask
            if (causal)
            {
                for (int b = 0; b < batch; b++)
                    for (int h = 0; h < numHeads; h++)
                        for (int i = 0; i < seqLen; i++)
                            for (int j = 0; j < seqLen; j++)
                                if (j > i)
                                    attnScores.data[(b * numHeads + h) * seqLen * seqLen + i * seqLen + j] = -1e9f;
            }

            // Mask addition
            if (mask != null)
            {
                for (int i = 0; i < attnScores.Length; i++)
                    attnScores.data[i] += mask.data[i % mask.Length];
            }

            // Softmax
            Tensor attnWeights = TensorOps.Softmax(attnScores, 2);

            // Output: attnWeights @ v
            Tensor output = TensorOps.MatMul(attnWeights, v);

            // Permute back: [batch, seqLen, numHeads, headDim]
            output = TensorOps.Permute(output, new[] { 0, 2, 1, 3 });

            // Reshape: [batch, seqLen, dModel]
            output = TensorOps.Reshape(output, new[] { batch, seqLen, dModel });

            // Output projection
            return oProj.Forward(output);
        }

        public override string ToString() => $"MultiHeadAttention(dModel={dModel}, heads={numHeads}, rope={useRope})";
    }

    /// <summary>
    /// SwiGLU feed-forward network (Llama-style).
    /// Mirrors Tensor-Engine's SwiGLU activation in transformer context.
    /// </summary>
    [Serializable]
    public class SwiGLUFFN
    {
        public Linear gateProj, upProj, downProj;
        public int dModel;
        public int dFF;

        public SwiGLUFFN(int dModel, int dFF)
        {
            this.dModel = dModel;
            this.dFF = dFF;
            gateProj = new Linear(dModel, dFF);
            upProj = new Linear(dModel, dFF);
            downProj = new Linear(dFF, dModel);
        }

        public Tensor Forward(Tensor input)
        {
            Tensor gate = gateProj.Forward(input);
            Tensor up = upProj.Forward(input);

            // SwiGLU: gate * sigmoid(gate) * up ... actually it's silu(gate) * up
            var siluGate = new float[gate.Length];
            for (int i = 0; i < gate.Length; i++)
            {
                float g = gate.data[i];
                siluGate[i] = g / (1f + Mathf.Exp(-g)) * up.data[i];
            }
            return downProj.Forward(new Tensor(siluGate, gate.shape));
        }

        public override string ToString() => $"SwiGLUFFN({dModel} -> {dFF} -> {dModel})";
    }

    /// <summary>
    /// Standard FFN (GELU-based). Mirrors Tensor-Engine's Linear1/Linear2 FFN.
    /// </summary>
    [Serializable]
    public class StandardFFN
    {
        public Linear linear1;
        public Linear linear2;
        public int dModel;
        public int dFF;

        public StandardFFN(int dModel, int dFF)
        {
            this.dModel = dModel;
            this.dFF = dFF;
            linear1 = new Linear(dModel, dFF);
            linear2 = new Linear(dFF, dModel);
        }

        public Tensor Forward(Tensor input)
        {
            Tensor h = linear1.Forward(input);
            h = TensorOps.Gelu(h);
            return linear2.Forward(h);
        }

        public override string ToString() => $"StandardFFN({dModel} -> {dFF} -> {dModel})";
    }

    /// <summary>
    /// A single Transformer block (attention + FFN with residual connections).
    /// Supports both standard and Llama-style (RMSNorm pre-norm, SwiGLU).
    /// Mirrors Tensor-Engine's TransformerBlock.
    /// </summary>
    [Serializable]
    public class TransformerBlock
    {
        public MultiHeadAttention mha;
        public StandardFFN ffn;
        public SwiGLUFFN swigluFfn;
        public LayerNorm attnNorm;
        public LayerNorm ffnNorm;
        public RMSNorm attnRmsNorm;
        public RMSNorm ffnRmsNorm;
        public int dModel;
        public int dFF;
        public int numHeads;
        public bool llamaStyle;
        public bool useRope;

        public TransformerBlock(int dModel, int dFF, int numHeads, bool llamaStyle = false, bool useRope = false)
        {
            this.dModel = dModel;
            this.dFF = dFF;
            this.numHeads = numHeads;
            this.llamaStyle = llamaStyle;
            this.useRope = useRope;

            mha = new MultiHeadAttention(dModel, numHeads, useRope);
            ffn = new StandardFFN(dModel, dFF);
            swigluFfn = new SwiGLUFFN(dModel, dFF);
            attnNorm = new LayerNorm(dModel);
            ffnNorm = new LayerNorm(dModel);
            attnRmsNorm = new RMSNorm(dModel);
            ffnRmsNorm = new RMSNorm(dModel);
        }

        /// <summary>
        /// Forward pass for standard transformer block.
        /// </summary>
        public Tensor Forward(Tensor input, bool causal = false)
        {
            if (llamaStyle)
            {
                return ForwardLlama(input, causal);
            }

            // Standard: x + attn(LN(x))
            Tensor normed = attnNorm.Forward(input);
            Tensor attnOut = mha.Forward(normed, causal);
            Tensor h = TensorOps.Add(input, attnOut);

            // x + mlp(LN(x))
            Tensor normed2 = ffnNorm.Forward(h);
            Tensor ffnOut = ffn.Forward(normed2);
            return TensorOps.Add(h, ffnOut);
        }

        /// <summary>
        /// Llama-style forward: RMSNorm pre-norm, SwiGLU FFN.
        /// </summary>
        public Tensor ForwardLlama(Tensor input, bool causal = false)
        {
            // Attention sub-block
            Tensor normed = attnRmsNorm.Forward(input);
            Tensor attnOut = mha.Forward(normed, causal);
            Tensor h = TensorOps.Add(input, attnOut);

            // FFN sub-block
            Tensor normed2 = ffnRmsNorm.Forward(h);
            Tensor ffnOut = swigluFfn.Forward(normed2);
            return TensorOps.Add(h, ffnOut);
        }

        public List<Linear> GetParameters()
        {
            var params_ = new List<Linear>();
            params_.Add(mha.qProj);
            params_.Add(mha.kProj);
            params_.Add(mha.vProj);
            params_.Add(mha.oProj);
            params_.Add(llamaStyle ? swigluFfn.gateProj : ffn.linear1);
            params_.Add(llamaStyle ? swigluFfn.upProj : ffn.linear1);
            params_.Add(llamaStyle ? swigluFfn.downProj : ffn.linear2);
            return params_;
        }

        public override string ToString() => $"TransformerBlock(dModel={dModel}, heads={numHeads}, ff={dFF}, llama={llamaStyle})";
    }

    /// <summary>
    /// Full Llama-style decoder model.
    /// Mirrors Tensor-Engine's Llama class / GPTDecoder.
    /// </summary>
    [Serializable]
    public class LlamaDecoder
    {
        public Embedding embedding;
        public TransformerBlock[] blocks;
        public RMSNorm outputNorm;
        public int dModel;
        public int dFF;
        public int numHeads;
        public int numLayers;
        public int vocabSize;
        public int headDim;
        public bool useRope;

        public LlamaDecoder(int vocabSize, int dModel, int dFF, int numHeads, int numLayers, int kvHeads, bool useRope = true)
        {
            this.vocabSize = vocabSize;
            this.dModel = dModel;
            this.dFF = dFF;
            this.numHeads = numHeads;
            this.numLayers = numLayers;
            this.useRope = useRope;
            headDim = dModel / numHeads;

            embedding = new Embedding(vocabSize, dModel);
            blocks = new TransformerBlock[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                blocks[i] = new TransformerBlock(dModel, dFF, numHeads, llamaStyle: true, useRope);
            }
            outputNorm = new RMSNorm(dModel);
        }

        /// <summary>
        /// Forward pass: token IDs -> logits.
        /// inputIds shape: [batch, seqLen]
        /// </summary>
        public Tensor Forward(Tensor inputIds)
        {
            int batch = inputIds.shape[0];
            int seqLen = inputIds.shape[1];

            // Embedding lookup: [batch, seqLen] -> [batch, seqLen, dModel]
            Tensor h = embedding.Forward(inputIds);

            // Transformer blocks
            for (int i = 0; i < numLayers; i++)
            {
                h = blocks[i].ForwardLlama(h, causal: true);
            }

            // Final RMSNorm
            h = outputNorm.Forward(h);

            // LM head: matmul with embedding weights (weight tying)
            // h: [batch, seqLen, dModel], embedding.weight: [vocab, dModel]
            // Output: [batch, seqLen, vocab]
            var logits = new float[batch * seqLen * vocabSize];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seqLen; s++)
                    for (int v = 0; v < vocabSize; v++)
                    {
                        float sum = 0f;
                        for (int d = 0; d < dModel; d++)
                            sum += h.data[(b * seqLen + s) * dModel + d] * embedding.Weight.data[v * dModel + d];
                        logits[(b * seqLen + s) * vocabSize + v] = sum;
                    }

            return new Tensor(logits, new[] { batch, seqLen, vocabSize });
        }

        /// <summary>
        /// Get all trainable parameters as Linear layers.
        /// </summary>
        public List<Linear> GetParameters()
        {
            var params_ = new List<Linear>();
            foreach (var blk in blocks) params_.AddRange(blk.GetParameters());
            return params_;
        }

        /// <summary>
        /// Forward with causal offset for multimodal: prepend image tokens before text tokens.
        /// </summary>
        public Tensor ForwardWithCausalOffset(Tensor input, int causalOffset)
        {
            int batch = input.shape[0];
            int seqLen = input.shape[1];
            Tensor h = input;

            for (int i = 0; i < numLayers; i++)
            {
                h = blocks[i].ForwardLlama(h, causal: true);
            }

            h = outputNorm.Forward(h);

            // LM head
            var logits = new float[batch * seqLen * vocabSize];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seqLen; s++)
                    for (int v = 0; v < vocabSize; v++)
                    {
                        float sum = 0f;
                        for (int d = 0; d < dModel; d++)
                            sum += h.data[(b * seqLen + s) * dModel + d] * embedding.Weight.data[v * dModel + d];
                        logits[(b * seqLen + s) * vocabSize + v] = sum;
                    }
            return new Tensor(logits, new[] { batch, seqLen, vocabSize });
        }

        public override string ToString() => $"LlamaDecoder(vocab={vocabSize}, dModel={dModel}, layers={numLayers}, heads={numHeads})";
    }

    /// <summary>
    /// Multimodal LLM: vision encoder + text decoder with projector.
    /// Mirrors Tensor-Engine's MultimodalLLM.
    /// </summary>
    [Serializable]
    public class MultimodalLLM
    {
        public VisionTransformer visionEncoder;
        public Linear projector;
        public LlamaDecoder textDecoder;
        public int dModel;
        public int visionDim;

        public MultimodalLLM(int visionDim, int dModel, int vocabSize, int dFF, int numHeads, int numLayers)
        {
            this.visionDim = visionDim;
            this.dModel = dModel;
            visionEncoder = new VisionTransformer(visionDim, dModel);
            projector = new Linear(dModel, dModel);
            textDecoder = new LlamaDecoder(vocabSize, dModel, dFF, numHeads, numLayers, numHeads);
        }

        /// <summary>
        /// Forward pass: image tensor + text token IDs -> logits.
        /// images: [batch, channels, height, width] as flat tensor
        /// inputIds: [batch, seqLen] token indices
        /// </summary>
        public Tensor Forward(Tensor images, Tensor inputIds)
        {
            // Vision encoding
            Tensor imgFeatures = visionEncoder.Forward(images); // [batch, num_patches, dModel]

            // Project to dModel
            int batch = imgFeatures.shape[0];
            Tensor imgProj = projector.Forward(imgFeatures);

            // Text embedding
            int seqLen = inputIds.shape[1];
            Tensor txtEmbed = textDecoder.embedding.Forward(inputIds);

            // Concatenate: [batch, num_patches + seqLen, dModel]
            int combinedLen = imgFeatures.shape[1] + seqLen;
            var combined = new float[batch * combinedLen * dModel];

            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < imgFeatures.shape[1]; i++)
                    for (int d = 0; d < dModel; d++)
                        combined[b * combinedLen * dModel + i * dModel + d] = imgProj.data[b * imgFeatures.shape[1] * dModel + i * dModel + d];

                for (int i = 0; i < seqLen; i++)
                    for (int d = 0; d < dModel; d++)
                        combined[b * combinedLen * dModel + (imgFeatures.shape[1] + i) * dModel + d] = txtEmbed.data[b * seqLen * dModel + i * dModel + d];
            }

            Tensor combinedTensor = new Tensor(combined, new[] { batch, combinedLen, dModel });

            // Decoder with causal offset = num_patches
            return textDecoder.ForwardWithCausalOffset(combinedTensor, imgFeatures.shape[1]);
        }

        public override string ToString() => $"MultimodalLLM(vision={visionDim}, dModel={dModel})";
    }

    /// <summary>
    /// Vision Transformer (ViT) for image encoding.
    /// Mirrors Tensor-Engine's VisionTransformer.
    /// </summary>
    [Serializable]
    public class VisionTransformer
    {
        public Conv2D patchEmbed;
        public LayerNorm preNorm;
        public TransformerBlock[] blocks;
        public LayerNorm postNorm;
        public int visionDim;
        public int dModel;
        public int imageDim;
        public int numPatches;

        public VisionTransformer(int imageDim, int dModel, int numHeads = 12, int numLayers = 12, int patchSize = 16)
        {
            this.visionDim = imageDim;
            this.dModel = dModel;
            this.imageDim = imageDim;

            // Patch embedding: Conv2D with kernel=patchSize, stride=patchSize
            patchEmbed = new Conv2D(3, dModel, patchSize, patchSize, 0);
            preNorm = new LayerNorm(dModel);

            blocks = new TransformerBlock[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                blocks[i] = new TransformerBlock(dModel, dModel * 4, numHeads);
            }
            postNorm = new LayerNorm(dModel);
        }

        /// <summary>
        /// Forward pass: image tensor -> [batch, num_patches, dModel]
        /// </summary>
        public Tensor Forward(Tensor images)
        {
            // images: [batch, channels, height, width]
            // For simplicity, use Conv2D to create patch embeddings
            Tensor patches = patchEmbed.Forward(images);
            // Reshape to [batch, num_patches, dModel]
            // This is a simplified version; real ViT needs more reshaping logic
            return patches;
        }

        public override string ToString() => $"VisionTransformer(imageDim={imageDim}, dModel={dModel})";
    }

    /// <summary>
    /// Simple Conv2D layer for patch embedding.
    /// </summary>
    [Serializable]
    public class Conv2D
    {
        public int inChannels, outChannels, kernelSize, stride, padding;
        public Linear linear; // Simplified: use linear to simulate conv

        public Conv2D(int inChannels, int outChannels, int kernelSize, int stride, int padding)
        {
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
            // Simplified: just store params, actual conv would need more work
            linear = new Linear(inChannels * kernelSize * kernelSize, outChannels);
        }

        public Tensor Forward(Tensor input)
        {
            // Simplified forward - just return input reshaped for now
            // Real Conv2D would iterate over patches
            return input;
        }

        public override string ToString() => $"Conv2D({inChannels}->{outChannels}, k={kernelSize}, s={stride})";
    }
}
