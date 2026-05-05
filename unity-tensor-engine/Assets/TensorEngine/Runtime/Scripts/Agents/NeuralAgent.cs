using System;
using System.Collections.Generic;
using UnityEngine;
using TensorEngine.Core;
using TensorEngine.Models;
using TensorEngine.Bridge;

namespace TensorEngine.Agents
{
    /// <summary>
    /// A simple tokenizer for Unity integration.
    /// Maps character indices to token IDs and back.
    /// In production, you'd use a real tokenizer (BPE, WordPiece, etc.)
    /// </summary>
    public class SimpleTokenizer
    {
        public Dictionary<string, int> vocab { get; private set; } = new Dictionary<string, int>();
        public int unkTokenId = 0;
        public int padTokenId = 1;
        public int bosTokenId = 2;
        public int eosTokenId = 3;

        public int VocabSize => vocab.Count;

        public SimpleTokenizer()
        {
            // Initialize with basic characters
            vocab["<unk>"] = unkTokenId;
            vocab["<pad>"] = padTokenId;
            vocab["<bos>"] = bosTokenId;
            vocab["<eos>"] = eosTokenId;

            // Add basic ASCII characters
            for (int i = 32; i < 127; i++)
            {
                vocab[Convert.ToChar(i).ToString()] = vocab.Count;
            }
            // Add common punctuation and whitespace
            vocab[" "] = vocab.Count;
            vocab["\n"] = vocab.Count;
            vocab["\t"] = vocab.Count;
            vocab["!"] = vocab.Count;
            vocab["?"] = vocab.Count;
            vocab[","] = vocab.Count;
            vocab["."] = vocab.Count;
            vocab[":"] = vocab.Count;
            vocab[";"] = vocab.Count;
            vocab["'"] = vocab.Count;
            vocab["\""] = vocab.Count;
            vocab["("] = vocab.Count;
            vocab[")"] = vocab.Count;
            vocab["-"] = vocab.Count;
            vocab["—"] = vocab.Count;
            vocab["…"] = vocab.Count;
        }

        public int Encode(string text)
        {
            int[] tokens = EncodeSequence(text);
            return tokens.Length > 0 ? tokens[0] : unkTokenId;
        }

        public int[] EncodeSequence(string text)
        {
            var tokens = new List<int>();
            tokens.Add(bosTokenId);

            // Simple character-level tokenization
            foreach (char c in text)
            {
                string key = c.ToString();
                if (vocab.ContainsKey(key))
                    tokens.Add(vocab[key]);
                else
                    tokens.Add(unkTokenId);
            }

            tokens.Add(eosTokenId);
            return tokens.ToArray();
        }

        public string Decode(int[] tokens)
        {
            var chars = new List<char>();
            foreach (int t in tokens)
            {
                foreach (var kvp in vocab)
                {
                    if (kvp.Value == t)
                    {
                        if (kvp.Key != "<bos>" && kvp.Key != "<eos>" && kvp.Key != "<pad>" && kvp.Key != "<unk>")
                            chars.Add(kvp.Key[0]);
                        break;
                    }
                }
            }
            return new string(chars.ToArray());
        }

        public string DecodeSingle(int token)
        {
            foreach (var kvp in vocab)
            {
                if (kvp.Value == token)
                    return kvp.Key;
            }
            return "?";
        }
    }

    /// <summary>
    /// Sampling strategies for text generation.
    /// </summary>
    public enum SamplingStrategy
    {
        Greedy,
        TopK,
        TopP,
        Temperature
    }

    /// <summary>
    /// Sampling parameters for text generation.
    /// </summary>
    [Serializable]
    public class SamplingParams
    {
        public SamplingStrategy strategy = SamplingStrategy.Temperature;
        public float temperature = 0.8f;
        public int topK = 50;
        public float topP = 0.95f;

        public int SampleFromLogits(Tensor logits)
        {
            if (logits == null || logits.Length == 0) return 0;

            float[] logProbs = logits.data;
            int vocabSize = logits.shape[logits.shape.Length - 1];

            // Get the last timestep logits
            int offset = logits.Length - vocabSize;
            float[] logits1d = new float[vocabSize];
            for (int i = 0; i < vocabSize; i++)
                logits1d[i] = logProbs[offset + i];

            switch (strategy)
            {
                case SamplingStrategy.Greedy:
                    return Argmax(logits1d);

                case SamplingStrategy.Temperature:
                    return SampleWithTemperature(logits1d, temperature);

                case SamplingStrategy.TopK:
                    return SampleTopK(logits1d, topK, temperature);

                case SamplingStrategy.TopP:
                    return SampleTopP(logits1d, topP, temperature);

                default:
                    return Argmax(logits1d);
            }
        }

        private int Argmax(float[] arr)
        {
            int maxIdx = 0;
            for (int i = 1; i < arr.Length; i++)
                if (arr[i] > arr[maxIdx]) maxIdx = i;
            return maxIdx;
        }

        private int SampleWithTemperature(float[] logits, float temp)
        {
            float scale = 1f / temp;
            float[] probs = new float[logits.Length];
            float maxLogit = logits[0];
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > maxLogit) maxLogit = logits[i];

            float sumExp = 0f;
            for (int i = 0; i < logits.Length; i++)
            {
                probs[i] = Mathf.Exp((logits[i] - maxLogit) * scale);
                sumExp += probs[i];
            }

            for (int i = 0; i < probs.Length; i++)
                probs[i] /= sumExp;

            return WeightedSample(probs);
        }

        private int SampleTopK(float[] logits, int k, float temp)
        {
            // Get top-k indices
            var indexed = new List<(int, float)>();
            for (int i = 0; i < logits.Length; i++)
                indexed.Add((i, logits[i]));
            indexed.Sort((a, b) => b.Item2.CompareTo(a.Item2));

            k = Mathf.Min(k, indexed.Count);
            var topKLogits = new float[k];
            var topKIndices = new int[k];
            for (int i = 0; i < k; i++)
            {
                topKLogits[i] = indexed[i].Item2;
                topKIndices[i] = indexed[i].Item1;
            }

            float scale = 1f / temp;
            float maxLogit = topKLogits[0];
            float sumExp = 0f;
            for (int i = 0; i < k; i++)
            {
                topKLogits[i] = Mathf.Exp((topKLogits[i] - maxLogit) * scale);
                sumExp += topKLogits[i];
            }

            var probs = new float[k];
            for (int i = 0; i < k; i++)
                probs[i] = topKLogits[i] / sumExp;

            int sampledIdx = WeightedSample(probs);
            return topKIndices[sampledIdx];
        }

        private int SampleTopP(float[] logits, float p, float temp)
        {
            var indexed = new List<(int, float)>();
            for (int i = 0; i < logits.Length; i++)
                indexed.Add((i, logits[i]));
            indexed.Sort((a, b) => b.Item2.CompareTo(a.Item2));

            float sumExp = 0f;
            float scale = 1f / temp;
            float maxLogit = indexed[0].Item2;
            var probs = new float[logits.Length];

            for (int i = 0; i < indexed.Count; i++)
            {
                probs[indexed[i].Item1] = Mathf.Exp((indexed[i].Item2 - maxLogit) * scale);
                sumExp += probs[indexed[i].Item1];
            }

            for (int i = 0; i < logits.Length; i++)
                probs[i] /= sumExp;

            // Cumulative sum
            float cumSum = 0f;
            int topPCount = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                cumSum += probs[indexed[i].Item1];
                topPCount++;
                if (cumSum >= p) break;
            }

            // Renormalize
            float renormSum = 0f;
            for (int i = 0; i < topPCount; i++)
                renormSum += probs[indexed[i].Item1];

            for (int i = 0; i < topPCount; i++)
                probs[indexed[i].Item1] /= renormSum;

            return WeightedSample(probs);
        }

        private int WeightedSample(float[] probs)
        {
            float r = UnityEngine.Random.value;
            float cumSum = 0f;
            for (int i = 0; i < probs.Length; i++)
            {
                cumSum += probs[i];
                if (r < cumSum) return i;
            }
            return probs.Length - 1;
        }
    }

    /// <summary>
    /// NeuralAgent: An AI-driven agent that uses Tensor-Engine models for decision making and behavior.
    /// Attach this to any GameObject to create a neural NPC.
    /// </summary>
    [RequireComponent(typeof(UnityEngine.Collider))]
    public class NeuralAgent : MonoBehaviour
    {
        [Header("Agent Identity")]
        [Tooltip("Agent name displayed in dialogue")]
        public string agentName = "Unknown";

        [Tooltip("Agent's personality description")]
        public string personality = "A friendly NPC";

        [Tooltip("Agent's current goal or task")]
        public string currentGoal = "Idle";

        [Header("Neural Model")]
        [Tooltip("Model ID in the Python bridge")]
        public string modelId = "npc_agent";

        [Tooltip("Path to the model file (SafeTensors)")]
        public string modelPath = "";

        [Tooltip("Model configuration path")]
        public string configPath = "";

        [Header("Behavior Parameters")]
        [Tooltip("Max tokens per response")]
        public int maxTokens = 200;

        [Tooltip("Sampling temperature")]
        public float temperature = 0.7f;

        [Tooltip("Top-K sampling parameter")]
        public int topK = 50;

        [Tooltip("Top-P sampling parameter")]
        public float topP = 0.95f;

        [Header("Dialogue Settings")]
        [Tooltip("Max dialogue history to send to model")]
        public int maxHistory = 10;

        [Tooltip("Show debug logs")]
        public bool debugMode = false;

        // Internal state
        private SimpleTokenizer tokenizer;
        private LlamaDecoder model;
        private bool isModelLoaded = false;
        private List<string> dialogueHistory = new List<string>();
        private List<int> tokenHistory = new List<int>();
        private SamplingParams samplingParams;
        private UnityEngine.Collider agentCollider;

        // Events
        public event System.Action<string> OnDialogueGenerated;
        public event System.Action<string> OnActionGenerated;
        public event System.Action OnModelLoaded;
        public event System.Action OnModelError;

        void Awake()
        {
            tokenizer = new SimpleTokenizer();
            samplingParams = new SamplingParams();
            samplingParams.temperature = temperature;
            samplingParams.topK = topK;
            samplingParams.topP = topP;
            agentCollider = GetComponent<UnityEngine.Collider>();
        }

        void Start()
        {
            if (PythonBridgeService.Instance == null)
            {
                Debug.LogError("[NeuralAgent] No PythonBridgeService found. Ensure one exists in the scene.");
                return;
            }

            // Try to load model asynchronously
            if (!string.IsNullOrEmpty(modelPath))
            {
                StartCoroutine(LoadModelAsync());
            }
        }

        /// <summary>
        /// Load the neural model from the specified path.
        /// </summary>
        public System.Collections.IEnumerator LoadModelAsync()
        {
            isModelLoaded = false;
            Debug.Log($"[NeuralAgent] Loading model: {modelPath}");

            var loadTask = PythonBridgeService.Instance.LoadModelAsync(modelId, modelPath);
            yield return new WaitUntil(() => loadTask.IsCompleted);

            if (loadTask.Result)
            {
                isModelLoaded = true;
                Debug.Log($"[NeuralAgent] Model loaded successfully: {modelId}");
                OnModelLoaded?.Invoke();
            }
            else
            {
                Debug.LogError($"[NeuralAgent] Failed to load model: {modelId}");
                OnModelError?.Invoke();
            }
        }

        /// <summary>
        /// Generate a dialogue response based on the input text.
        /// </summary>
        public System.Collections.IEnumerator GenerateDialogue(string input, System.Action<string> onResult = null)
        {
            if (!isModelLoaded)
            {
                Debug.LogWarning("[NeuralAgent] Model not loaded. Cannot generate dialogue.");
                onResult?.Invoke($"[Error: Model not loaded]");
                yield break;
            }

            // Build prompt with history
            string prompt = BuildDialoguePrompt(input);

            if (debugMode)
                Debug.Log($"[NeuralAgent] Prompt: {prompt}");

            // Encode input
            int[] tokens = tokenizer.EncodeSequence(prompt);

            // Convert to tensor
            var inputTensor = new Tensor(tokens, new[] { 1, tokens.Length });

            // Run inference
            var inferenceTask = PythonBridgeService.Instance.InferenceAsync(modelId, inputTensor, maxTokens, temperature);
            yield return new WaitUntil(() => inferenceTask.IsCompleted);

            if (inferenceTask.Result != null)
            {
                // Sample the output
                int sampledToken = samplingParams.SampleFromLogits(inferenceTask.Result);
                string response = tokenizer.DecodeSingle(sampledToken);

                // Update history
                dialogueHistory.Add(input);
                dialogueHistory.Add(response);
                tokenHistory.AddRange(tokens);
                tokenHistory.Add(sampledToken);

                // Keep history within bounds
                if (dialogueHistory.Count > maxHistory * 2)
                {
                    dialogueHistory = dialogueHistory.GetRange(dialogueHistory.Count - maxHistory * 2, maxHistory * 2);
                    tokenHistory = tokenHistory.GetRange(tokenHistory.Count - maxHistory * 2, maxHistory * 2);
                }

                OnDialogueGenerated?.Invoke(response);
                onResult?.Invoke(response);

                if (debugMode)
                    Debug.Log($"[NeuralAgent] {agentName}: {response}");
            }
            else
            {
                string error = "[Error: Inference failed]";
                onResult?.Invoke(error);
            }
        }

        /// <summary>
        /// Generate an action for the agent to perform.
        /// </summary>
        public System.Collections.IEnumerator GenerateAction(System.Action<string> onResult = null)
        {
            if (!isModelLoaded)
            {
                Debug.LogWarning("[NeuralAgent] Model not loaded. Cannot generate action.");
                onResult?.Invoke("[Error: Model not loaded]");
                yield break;
            }

            string actionPrompt = BuildActionPrompt();

            int[] tokens = tokenizer.EncodeSequence(actionPrompt);
            var inputTensor = new Tensor(tokens, new[] { 1, tokens.Length });

            var inferenceTask = PythonBridgeService.Instance.InferenceAsync(modelId, inputTensor, maxTokens / 2, temperature);
            yield return new WaitUntil(() => inferenceTask.IsCompleted);

            if (inferenceTask.Result != null)
            {
                int sampledToken = samplingParams.SampleFromLogits(inferenceTask.Result);
                string action = tokenizer.DecodeSingle(sampledToken);

                OnActionGenerated?.Invoke(action);
                onResult?.Invoke(action);

                if (debugMode)
                    Debug.Log($"[NeuralAgent] Action: {action}");
            }
        }

        /// <summary>
        /// Build a dialogue prompt with context.
        /// </summary>
        private string BuildDialoguePrompt(string input)
        {
            string history = "";
            if (dialogueHistory.Count > 0)
            {
                history = "Previous conversation:\n";
                for (int i = Math.Max(0, dialogueHistory.Count - maxHistory * 2); i < dialogueHistory.Count; i++)
                {
                    history += $"{dialogueHistory[i]}\n";
                }
                history += "\n";
            }

            return $"""
=== NPC Dialogue System ===
Agent: {agentName}
Personality: {personality}
Current Goal: {currentGoal}

{history}
Player: {input}
{agentName}:
""";
        }

        /// <summary>
        /// Build an action prompt based on the agent's state.
        /// </summary>
        private string BuildActionPrompt()
        {
            string nearbyInfo = "No nearby entities detected.";
            if (agentCollider != null && agentCollider.bounds.size.magnitude > 0)
            {
                nearbyInfo = $"Detection radius: {agentCollider.bounds.size.magnitude}m";
            }

            return $"""
=== NPC Action Generator ===
Agent: {agentName}
Current Goal: {currentGoal}
Nearby: {nearbyInfo}
Environment: {GetEnvironmentDescription()}

Generate an action for {agentName} to perform:
""";
        }

        /// <summary>
        /// Get a brief description of the surrounding environment.
        /// </summary>
        private string GetEnvironmentDescription()
        {
            // Simple environment description based on nearby objects
            var nearbyObjects = UnityEngine.Object.FindObjectsOfType<UnityEngine.GameObject>();
            int count = 0;
            foreach (var obj in nearbyObjects)
            {
                if (obj != gameObject && Vector3.Distance(obj.transform.position, transform.position) < 10f)
                {
                    count++;
                }
            }
            return $"{count} nearby objects";
        }

        /// <summary>
        /// Clear the dialogue history.
        /// </summary>
        public void ClearHistory()
        {
            dialogueHistory.Clear();
            tokenHistory.Clear();
            if (debugMode)
                Debug.Log($"[NeuralAgent] History cleared for {agentName}");
        }

        /// <summary>
        /// Get the current dialogue history as a string.
        /// </summary>
        public string GetHistoryString()
        {
            return string.Join("\n", dialogueHistory);
        }

        /// <summary>
        /// Check if the model is loaded and ready.
        /// </summary>
        public bool IsReady() => isModelLoaded;

        void OnDrawGizmosSelected()
        {
            // Visualize agent's detection radius
            Gizmos.color = isModelLoaded ? Color.green : Color.red;
            Gizmos.DrawWireSphere(transform.position, 5f);
            Gizmos.color = isModelLoaded ? Color.green : Color.red;
            Gizmos.DrawWireSphere(transform.position, 5f);
        }
    }

    /// <summary>
    /// NeuralBehavior: A behavior tree-like system that uses neural models for decision making.
    /// Provides a more structured approach to neural NPC behavior.
    /// </summary>
    public class NeuralBehavior : MonoBehaviour
    {
        [Header("Behavior Configuration")]
        [Tooltip("List of possible behaviors")]
        public List<BehaviorNode> behaviors = new List<BehaviorNode>();

        [Tooltip("Current active behavior")]
        public BehaviorNode currentBehavior;

        [Tooltip("Behavior update interval (seconds)")]
        public float updateInterval = 1f;

        [Tooltip("Use neural model for behavior selection")]
        public bool useNeuralSelection = true;

        [Tooltip("Neural agent for behavior generation")]
        public NeuralAgent neuralAgent;

        // Internal
        private float nextUpdate = 0f;
        private UnityEngine.Collider agentCollider;

        void Awake()
        {
            agentCollider = GetComponent<UnityEngine.Collider>();
            if (behaviors.Count > 0)
                currentBehavior = behaviors[0];
        }

        void Update()
        {
            if (Time.time >= nextUpdate)
            {
                UpdateBehavior();
                nextUpdate = Time.time + updateInterval;
            }
        }

        /// <summary>
        /// Update the current behavior.
        /// </summary>
        private void UpdateBehavior()
        {
            if (useNeuralSelection && neuralAgent != null && neuralAgent.IsReady())
            {
                // Use neural model to select behavior
                string context = BuildBehaviorContext();
                neuralAgent.GenerateDialogue($"Choose a behavior based on: {context}", OnBehaviorSelected);
            }
            else if (currentBehavior != null)
            {
                currentBehavior.Execute(this);
            }
        }

        /// <summary>
        /// Build context for behavior selection.
        /// </summary>
        private string BuildBehaviorContext()
        {
            string nearby = "nothing nearby";
            if (agentCollider != null)
            {
                var objects = UnityEngine.Object.FindObjectsOfType<UnityEngine.GameObject>();
                int count = 0;
                foreach (var obj in objects)
                {
                    if (obj != gameObject && Vector3.Distance(obj.transform.position, transform.position) < 10f)
                        count++;
                }
                nearby = $"{count} objects nearby";
            }

            return $"Player is nearby ({nearby}). Current goal: {neuralAgent != null ? neuralAgent.currentGoal : "none"}. Time: {System.DateTime.Now:HH:mm}. Weather: Clear.";
        }

        /// <summary>
        /// Callback for neural behavior selection.
        /// </summary>
        private void OnBehaviorSelected(string response)
        {
            // Parse the response and select the appropriate behavior
            foreach (var behavior in behaviors)
            {
                if (response.Contains(behavior.name.ToLower()))
                {
                    currentBehavior = behavior;
                    if (neuralAgent != null && neuralAgent.debugMode)
                        Debug.Log($"[NeuralBehavior] Selected: {behavior.name}");
                    break;
                }
            }

            if (currentBehavior != null)
                currentBehavior.Execute(this);
        }

        /// <summary>
        /// Set a new behavior.
        /// </summary>
        public void SetBehavior(BehaviorNode behavior)
        {
            currentBehavior = behavior;
        }

        /// <summary>
        /// Add a behavior to the list.
        /// </summary>
        public void AddBehavior(BehaviorNode behavior)
        {
            behaviors.Add(behavior);
        }
    }

    /// <summary>
    /// A single behavior node in the neural behavior system.
    /// </summary>
    [Serializable]
    public class BehaviorNode
    {
        public string name = "Behavior";
        public string description = "A behavior node";
        public UnityEngine.Sprite icon;
        public Color color = Color.white;

        public System.Action<NeuralBehavior> onEnter;
        public System.Action<NeuralBehavior> onExecute;
        public System.Action<NeuralBehavior> onExit;

        public bool isActive = false;

        public void Execute(NeuralBehavior behavior)
        {
            if (!isActive)
            {
                isActive = true;
                onEnter?.Invoke(behavior);
            }
            onExecute?.Invoke(behavior);
        }

        public void Exit(NeuralBehavior behavior)
        {
            isActive = false;
            onExit?.Invoke(behavior);
        }
    }
}
