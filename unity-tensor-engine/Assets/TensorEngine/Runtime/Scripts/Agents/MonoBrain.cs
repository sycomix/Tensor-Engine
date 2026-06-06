using System;
using System.Collections.Generic;
using UnityEngine;
using TensorEngine.Core;
using TensorEngine.Models;
using TensorEngine.Bridge;
using TensorEngine.Agents;

namespace TensorEngine
{
    /// <summary>
    /// MonoBrain: The central brain component that manages neural models, agents, and the Python bridge.
    /// Attach one to a GameManager or SceneController object.
    /// </summary>
    public class MonoBrain : MonoBehaviour
    {
        public static MonoBrain Instance { get; private set; }

        [Header("Model Management")]
        [Tooltip("List of available models")]
        public List<NeuralModelConfig> availableModels = new List<NeuralModelConfig>();

        [Tooltip("Currently active model")]
        public NeuralModelConfig activeModel;

        [Tooltip("Auto-load the active model on start")]
        public bool autoLoadModel = true;

        [Header("Agent Management")]
        [Tooltip("List of registered neural agents")]
        public List<NeuralAgent> registeredAgents = new List<NeuralAgent>();

        [Tooltip("Max concurrent inference requests")]
        public int maxConcurrentRequests = 5;

        [Header("Performance")]
        [Tooltip("Enable GPU acceleration if available")]
        public bool enableGPU = false;

        [Tooltip("Batch size for inference")]
        public int batchSize = 1;

        [Header("Logging")]
        [Tooltip("Enable detailed logging")]
        public bool verboseLogging = false;

        // Internal state
        private PythonBridgeService bridgeService;
        private Dictionary<string, LlamaDecoder> modelCache = new Dictionary<string, LlamaDecoder>();
        private Queue<InferenceRequest> inferenceQueue = new Queue<InferenceRequest>();
        private int activeRequests = 0;

        // Events
        public event System.Action<NeuralModelConfig> OnModelLoaded;
        public event System.Action<string> OnModelError;
        public event System.Action<NeuralAgent, string> OnAgentDialogue;

        void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            bridgeService = FindFirstObjectByType<PythonBridgeService>();
            if (bridgeService == null)
            {
                Debug.LogError("[MonoBrain] No PythonBridgeService found. Creating one...");
                var go = new GameObject("PythonBridgeService");
                bridgeService = go.AddComponent<PythonBridgeService>();
                bridgeService.autoStartServer = true;
            }
        }

        void Start()
        {
            // Register all NeuralAgents in the scene
            var agents = FindObjectsByType<NeuralAgent>(FindObjectsSortMode.None);
            foreach (var agent in agents)
            {
                if (!registeredAgents.Contains(agent))
                    registeredAgents.Add(agent);
            }

            if (autoLoadModel && activeModel != null)
            {
                LoadModel(activeModel);
            }
        }

        /// <summary>
        /// Load a model into the Python bridge.
        /// </summary>
        public System.Collections.IEnumerator LoadModel(NeuralModelConfig config)
        {
            if (config == null || string.IsNullOrEmpty(config.modelPath))
            {
                Debug.LogError("[MonoBrain] Invalid model config.");
                yield break;
            }

            Debug.Log($"[MonoBrain] Loading model: {config.name} from {config.modelPath}");

            var loadTask = bridgeService.LoadModelAsync(config.modelId, config.modelPath);
            yield return new WaitUntil(() => loadTask.IsCompleted);

            if (loadTask.Result)
            {
                modelCache[config.modelId] = config.CreateModel();
                activeModel = config;
                Debug.Log($"[MonoBrain] Model loaded: {config.name}");
                OnModelLoaded?.Invoke(config);
            }
            else
            {
                Debug.LogError($"[MonoBrain] Failed to load model: {config.name}");
                OnModelError?.Invoke(config.name);
            }
        }

        /// <summary>
        /// Queue an inference request.
        /// </summary>
        public void QueueInference(string modelId, Tensor input, System.Action<Tensor> onComplete, float priority = 0f)
        {
            if (activeRequests >= maxConcurrentRequests)
            {
                inferenceQueue.Enqueue(new InferenceRequest(modelId, input, onComplete, priority));
                if (verboseLogging)
                    Debug.Log($"[MonoBrain] Inference queued. Queue size: {inferenceQueue.Count}");
                return;
            }

            activeRequests++;
            StartInference(modelId, input, onComplete);
        }

        private void StartInference(string modelId, Tensor input, System.Action<Tensor> onComplete)
        {
            var task = bridgeService.InferenceAsync(modelId, input);
            StartCoroutine(WaitForInference(task, onComplete));
        }

        private System.Collections.IEnumerator WaitForInference(System.Threading.Tasks.Task<Tensor> task, System.Action<Tensor> onComplete)
        {
            yield return new WaitUntil(() => task.IsCompleted);

            activeRequests--;

            if (task.Result != null)
            {
                onComplete?.Invoke(task.Result);
            }
            else
            {
                OnModelError?.Invoke("Inference failed");
            }

            // Process next queued request
            if (inferenceQueue.Count > 0)
            {
                var next = inferenceQueue.Dequeue();
                StartInference(next.modelId, next.input, next.onComplete);
            }
        }

        /// <summary>
        /// Generate dialogue for a specific agent.
        /// </summary>
        public System.Collections.IEnumerator GenerateAgentDialogue(NeuralAgent agent, string input, System.Action<string> onResult)
        {
            if (!agent.IsReady())
            {
                onResult?.Invoke("[Error: Agent model not loaded]");
                yield break;
            }

            var task = agent.GenerateDialogue(input, onResult);
            yield return task;

            OnAgentDialogue?.Invoke(agent, onResult != null ? "done" : "");
        }

        /// <summary>
        /// Get a tensor operation result from the Python server.
        /// </summary>
        public System.Collections.IEnumerator ComputeTensor(Tensor input, string operation, System.Action<Tensor> onResult, Tensor[] inputs = null, float[] floatArgs = null)
        {
            var task = bridgeService.ComputeAsync(operation, input, inputs, floatArgs);
            yield return new WaitUntil(() => task.IsCompleted);

            if (task.Result != null)
                onResult?.Invoke(task.Result);
        }

        /// <summary>
        /// Unload a model from memory.
        /// </summary>
        public void UnloadModel(string modelId)
        {
            if (modelCache.ContainsKey(modelId))
            {
                modelCache.Remove(modelId);
                Debug.Log($"[MonoBrain] Model unloaded: {modelId}");
            }
        }

        /// <summary>
        /// Get all loaded models.
        /// </summary>
        public string[] GetLoadedModels()
        {
            return bridgeService.ListModelsAsync().Result;
        }

        /// <summary>
        /// Register an agent.
        /// </summary>
        public void RegisterAgent(NeuralAgent agent)
        {
            if (!registeredAgents.Contains(agent))
                registeredAgents.Add(agent);
        }

        /// <summary>
        /// Unregister an agent.
        /// </summary>
        public void UnregisterAgent(NeuralAgent agent)
        {
            registeredAgents.Remove(agent);
        }

        void OnDestroy()
        {
            registeredAgents.Clear();
        }

        void OnGUI()
        {
            if (!verboseLogging) return;

            GUILayout.BeginArea(new Rect(10, 10, 300, 200));
            GUILayout.Label($"[MonoBrain Debug]");
            GUILayout.Label($"Active Models: {modelCache.Count}");
            GUILayout.Label($"Registered Agents: {registeredAgents.Count}");
            GUILayout.Label($"Active Requests: {activeRequests}");
            GUILayout.Label($"Queued Requests: {inferenceQueue.Count}");
            GUILayout.EndArea();
        }
    }

    /// <summary>
    /// Configuration for a neural model.
    /// </summary>
    [Serializable]
    public class NeuralModelConfig
    {
        public string name = "Unnamed Model";
        public string modelId = "default";
        public string modelPath = "";
        public string configPath = "";
        public string description = "";
        public UnityEngine.Sprite icon;
        public Color color = Color.white;

        // Model parameters
        public int dModel = 768;
        public int dFF = 3072;
        public int numHeads = 12;
        public int numLayers = 12;
        public int vocabSize = 50000;
        public bool useRope = true;

        // Runtime parameters
        public float defaultTemperature = 0.7f;
        public int defaultTopK = 50;
        public float defaultTopP = 0.95f;

        // Create a model instance
        public LlamaDecoder CreateModel()
        {
            return new LlamaDecoder(vocabSize, dModel, dFF, numHeads, numLayers, numHeads, useRope);
        }
    }

    /// <summary>
    /// Internal class for queued inference requests.
    /// </summary>
    [Serializable]
    public class InferenceRequest
    {
        public string modelId;
        public Tensor input;
        public System.Action<Tensor> onComplete;
        public float priority;

        public InferenceRequest(string modelId, Tensor input, System.Action<Tensor> onComplete, float priority)
        {
            this.modelId = modelId;
            this.input = input;
            this.onComplete = onComplete;
            this.priority = priority;
        }
    }

    /// <summary>
    /// NeuralEnvironment: Manages the neural context of a scene (weather, time, nearby agents).
    /// </summary>
    public class NeuralEnvironment : MonoBehaviour
    {
        [Header("Environment State")]
        public string timeOfDay = "Morning";
        public string weather = "Clear";
        public string season = "Spring";
        public float ambientMood = 0.5f; // 0 = dark, 1 = bright

        [Header("Dynamic Events")]
        public List<NeuralEvent> activeEvents = new List<NeuralEvent>();

        [Header("Sensors")]
        public float nearbyAgentCount;
        public float playerProximity;

        void Update()
        {
            // Update sensors
            var agents = FindObjectsByType<NeuralAgent>(FindObjectsSortMode.None);
            nearbyAgentCount = agents.Length;

            var player = GameObject.FindGameObjectWithTag("Player");
            if (player != null)
            {
                playerProximity = Vector3.Distance(player.transform.position, transform.position);
            }
        }

        /// <summary>
        /// Add a dynamic event to the environment.
        /// </summary>
        public void AddEvent(NeuralEvent @event)
        {
            activeEvents.Add(@event);
            if (MonoBrain.Instance != null && MonoBrain.Instance.verboseLogging)
                Debug.Log($"[NeuralEnvironment] Event added: {@event.name}");
        }

        /// <summary>
        /// Remove an event.
        /// </summary>
        public void RemoveEvent(NeuralEvent @event)
        {
            activeEvents.Remove(@event);
        }

        /// <summary>
        /// Get a context string for neural agents.
        /// </summary>
        public string GetContextString()
        {
            return $"Time: {timeOfDay}, Weather: {weather}, Season: {season}, Mood: {ambientMood:F2}, Events: {activeEvents.Count}";
        }
    }

    /// <summary>
    /// A dynamic event in the environment.
    /// </summary>
    [Serializable]
    public class NeuralEvent
    {
        public string name = "Event";
        public string description = "A dynamic event";
        public float duration = 60f;
        public float intensity = 0.5f;
        public bool isActive = false;

        public void Activate() => isActive = true;
        public void Deactivate() => isActive = false;
    }

    /// <summary>
    /// NeuralMemory: A persistent memory system for neural agents.
    /// Stores experiences, facts, and learned behaviors.
    /// </summary>
    public class NeuralMemory : MonoBehaviour
    {
        [Header("Memory Configuration")]
        [Tooltip("Max memory entries")]
        public int maxEntries = 1000;

        [Tooltip("Memory decay rate (0-1, lower = faster decay)")]
        public float decayRate = 0.95f;

        [Tooltip("Enable semantic search")]
        public bool enableSemanticSearch = true;

        // Internal
        private List<MemoryEntry> memories = new List<MemoryEntry>();
        private Dictionary<string, float> factCache = new Dictionary<string, float>();

        void Awake()
        {
            if (MonoBrain.Instance != null && MonoBrain.Instance.verboseLogging)
                Debug.Log("[NeuralMemory] Initialized with max entries: " + maxEntries);
        }

        /// <summary>
        /// Store a memory entry.
        /// </summary>
        public void StoreMemory(string agentId, string content, float importance = 0.5f)
        {
            if (memories.Count >= maxEntries)
            {
                // Remove oldest entry
                memories.RemoveAt(0);
            }

            memories.Add(new MemoryEntry(agentId, content, importance, System.DateTime.Now));

            // Update fact cache
            factCache[content.ToLower()] = importance;

            if (MonoBrain.Instance != null && MonoBrain.Instance.verboseLogging)
                Debug.Log($"[NeuralMemory] Stored: {content.Substring(0, Mathf.Min(50, content.Length))}...");
        }

        /// <summary>
        /// Retrieve memories for an agent.
        /// </summary>
        public List<string> RetrieveMemories(string agentId, int limit = 10)
        {
            var agentMemories = new List<string>();
            foreach (var m in memories)
            {
                if (m.agentId == agentId)
                    agentMemories.Add(m.content);
                if (agentMemories.Count >= limit) break;
            }
            return agentMemories;
        }

        /// <summary>
        /// Search memories by keyword.
        /// </summary>
        public List<string> SearchMemories(string keyword)
        {
            var results = new List<string>();
            keyword = keyword.ToLower();
            foreach (var m in memories)
            {
                if (m.content.ToLower().Contains(keyword))
                    results.Add(m.content);
            }
            return results;
        }

        /// <summary>
        /// Clear all memories.
        /// </summary>
        public void ClearMemories()
        {
            memories.Clear();
            factCache.Clear();
        }

        /// <summary>
        /// Get memory count.
        /// </summary>
        public int GetMemoryCount(string agentId = null)
        {
            if (agentId == null) return memories.Count;
            int count = 0;
            foreach (var m in memories)
                if (m.agentId == agentId) count++;
            return count;
        }
    }

    /// <summary>
    /// A single memory entry.
    /// </summary>
    [Serializable]
    public class MemoryEntry
    {
        public string agentId;
        public string content;
        public float importance;
        public System.DateTime timestamp;
        public float lastAccessed;
        public int accessCount;

        public MemoryEntry(string agentId, string content, float importance, System.DateTime timestamp)
        {
            this.agentId = agentId;
            this.content = content;
            this.importance = importance;
            this.timestamp = timestamp;
            this.lastAccessed = 0f;
            this.accessCount = 0;
        }
    }
}
