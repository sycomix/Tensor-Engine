using System;
using System.Collections.Generic;
using UnityEngine;
using TensorEngine.Bridge;
using TensorEngine.Agents;

namespace TensorEngine
{
    /// <summary>
    /// MonoBrain: Central manager that coordinates the Rust engine and neural agents.
    /// Attach one to a GameManager or SceneController object.
    /// </summary>
    public class MonoBrain : MonoBehaviour
    {
        public static MonoBrain Instance { get; private set; }

        [Header("Model Management")]
        [Tooltip("Automatically discover available models from the engine")]
        public bool autoDiscoverModels = true;

        [Tooltip("Currently active model ID")]
        public string activeModelId = "";

        [Tooltip("Available model IDs (populated from engine)")]
        public List<string> availableModelIds = new List<string>();

        [Header("Agent Management")]
        [Tooltip("List of registered neural agents")]
        public List<NeuralAgent> registeredAgents = new List<NeuralAgent>();

        [Header("Bridge Settings")]
        [Tooltip("Path to engine.exe")]
        public string enginePath = "engine.exe";

        [Tooltip("Path to the model directory")]
        public string modelPath = "";

        [Tooltip("Engine server port")]
        public int enginePort = 9090;

        [Tooltip("Auto-start the engine on Awake")]
        public bool autoStartEngine = true;

        [Header("Logging")]
        [Tooltip("Enable detailed logging")]
        public bool verboseLogging = false;

        private PythonBridgeService bridgeService;

        public event Action<string> OnModelDiscovered;
        public event Action OnEngineReady;
        public event Action<string> OnEngineError;

        void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        void Start()
        {
            // Find or create PythonBridgeService
            bridgeService = FindFirstObjectByType<PythonBridgeService>();
            if (bridgeService == null)
            {
                var go = new GameObject("PythonBridgeService");
                bridgeService = go.AddComponent<PythonBridgeService>();
            }

            // Configure the bridge
            bridgeService.enginePath = enginePath;
            bridgeService.modelPath = modelPath;
            bridgeService.serverPort = enginePort;
            bridgeService.autoStartEngine = autoStartEngine;

            // Register all NeuralAgents in the scene
            var agents = FindObjectsByType<NeuralAgent>(FindObjectsSortMode.None);
            foreach (var agent in agents)
            {
                if (!registeredAgents.Contains(agent))
                    registeredAgents.Add(agent);
            }

            // Listen for engine readiness to discover models
            if (autoDiscoverModels)
                StartCoroutine(DiscoverModelsWhenReady());
        }

        private System.Collections.IEnumerator DiscoverModelsWhenReady()
        {
            // Wait for bridge to be running
            yield return new WaitUntil(() => bridgeService.IsRunning);

            yield return new WaitForSeconds(1);

            // Discover models from the engine
            var task = bridgeService.ListModelsAsync();
            yield return new WaitUntil(() => task.IsCompleted);

            if (task.Result != null && task.Result.Length > 0)
            {
                availableModelIds.Clear();
                foreach (var id in task.Result)
                {
                    availableModelIds.Add(id);
                    if (verboseLogging)
                        Debug.Log($"[MonoBrain] Discovered model: {id}");
                    OnModelDiscovered?.Invoke(id);
                }

                if (string.IsNullOrEmpty(activeModelId) && availableModelIds.Count > 0)
                    activeModelId = availableModelIds[0];

                OnEngineReady?.Invoke();
            }
            else
            {
                Debug.LogWarning("[MonoBrain] No models discovered from engine.");
                OnEngineError?.Invoke("No models available");
            }
        }

        /// <summary>
        /// Generate dialogue for a specific agent.
        /// </summary>
        public void GenerateAgentDialogue(NeuralAgent agent, string input)
        {
            if (!registeredAgents.Contains(agent))
            {
                Debug.LogWarning($"[MonoBrain] Agent {agent.agentName} not registered.");
                return;
            }

            agent.GenerateDialogue(input);
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

        /// <summary>
        /// Check if the engine is running and ready.
        /// </summary>
        public bool IsEngineReady() => bridgeService != null && bridgeService.IsRunning;

        void OnDestroy()
        {
            registeredAgents.Clear();
        }

        void OnGUI()
        {
            if (!verboseLogging) return;

            GUILayout.BeginArea(new Rect(10, 10, 350, 300));
            GUILayout.Label("[MonoBrain Debug]");
            GUILayout.Label($"Engine Running: {IsEngineReady()}");
            GUILayout.Label($"Active Model: {activeModelId}");
            GUILayout.Label($"Available Models: {availableModelIds.Count}");
            GUILayout.Label($"Registered Agents: {registeredAgents.Count}");
            GUILayout.EndArea();
        }
    }
}
