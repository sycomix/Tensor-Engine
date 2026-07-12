using System;
using System.Collections.Generic;
using UnityEngine;
using TensorEngine.Bridge;

namespace TensorEngine.Agents
{
    [RequireComponent(typeof(UnityEngine.Collider))]
    public class NeuralAgent : MonoBehaviour
    {
        [Header("Agent Identity")]
        [Tooltip("Agent name displayed in dialogue")]
        public string agentName = "Unknown";

        [Tooltip("Agent's personality/backstory (system prompt)")]
        public string personality = "A friendly NPC";

        [Tooltip("Agent's current goal or task")]
        public string currentGoal = "Idle";

        [Header("Generation Parameters")]
        [Tooltip("Max tokens per response")]
        public int maxTokens = 200;

        [Tooltip("Sampling temperature")]
        public float temperature = 0.7f;

        [Tooltip("Top-P sampling parameter")]
        public float topP = 0.95f;

        [Header("Dialogue Settings")]
        [Tooltip("Max dialogue history pairs to include")]
        public int maxHistory = 10;

        [Tooltip("Show debug logs")]
        public bool debugMode = false;

        private List<ChatMessage> dialogueHistory = new List<ChatMessage>();
        private List<ChatMessage> actionContext = new List<ChatMessage>();
        private bool isStreaming = false;
        private string currentStreamId;
        private string accumulatedResponse = "";
        private UnityEngine.Collider agentCollider;
        private bool modelReadyNotified;

        public event Action<string> OnDialogueGenerated;
        public event Action<string> OnDialogueToken;
        public event Action<string> OnActionGenerated;
        public event Action OnModelReady;

        public bool IsReady() => PythonBridgeService.Instance != null && PythonBridgeService.Instance.IsRunning;

        private System.Collections.IEnumerator NotifyWhenModelReady()
        {
            yield return new WaitUntil(IsReady);
            if (!modelReadyNotified)
            {
                modelReadyNotified = true;
                OnModelReady?.Invoke();
            }
        }

        void Awake()
        {
            agentCollider = GetComponent<UnityEngine.Collider>();
        }

        void Start()
        {
            if (PythonBridgeService.Instance == null)
            {
                Debug.LogError("[NeuralAgent] No PythonBridgeService found.");
            }
            else
            {
                StartCoroutine(NotifyWhenModelReady());
            }
        }

        void OnDisable()
        {
            if (isStreaming && !string.IsNullOrEmpty(currentStreamId))
            {
                PythonBridgeService.Instance?.CancelStream(currentStreamId);
                isStreaming = false;
            }
        }

        /// <summary>
        /// Generate a dialogue response using the OpenAI chat format.
        /// Streams tokens via OnDialogueToken, final response via OnDialogueGenerated.
        /// </summary>
        public void GenerateDialogue(string input)
        {
            if (!IsReady())
            {
                Debug.LogWarning("[NeuralAgent] Engine not ready.");
                OnDialogueGenerated?.Invoke("[Error: Engine not ready]");
                return;
            }

            if (isStreaming)
            {
                Debug.LogWarning("[NeuralAgent] Already generating. Ignoring request.");
                return;
            }

            var messages = new List<ChatMessage>();
            messages.Add(new ChatMessage("system",
                $"You are {agentName}. Personality: {personality}. Current goal: {currentGoal}. Respond in character."));

            int startIdx = Math.Max(0, dialogueHistory.Count - maxHistory * 2);
            for (int i = startIdx; i < dialogueHistory.Count; i++)
                messages.Add(dialogueHistory[i]);

            messages.Add(new ChatMessage("user", input));
            dialogueHistory.Add(new ChatMessage("user", input));

            accumulatedResponse = "";
            isStreaming = true;

            currentStreamId = PythonBridgeService.Instance.StartChatCompletion(
                messages,
                onToken: (token) =>
                {
                    accumulatedResponse += token;
                    OnDialogueToken?.Invoke(token);
                },
                onComplete: () =>
                {
                    dialogueHistory.Add(new ChatMessage("assistant", accumulatedResponse));
                    OnDialogueGenerated?.Invoke(accumulatedResponse);
                    if (debugMode)
                        Debug.Log($"[NeuralAgent] {agentName}: {accumulatedResponse}");
                    isStreaming = false;
                },
                onError: (err) =>
                {
                    Debug.LogError($"[NeuralAgent] Generation error: {err}");
                    OnDialogueGenerated?.Invoke($"[Error: {err}]");
                    isStreaming = false;
                },
                temperature: temperature,
                topP: topP,
                maxTokens: maxTokens
            );
        }

        /// <summary>
        /// Generate an action for the agent using the completions endpoint.
        /// </summary>
        public void GenerateAction()
        {
            if (!IsReady())
            {
                Debug.LogWarning("[NeuralAgent] Engine not ready.");
                OnActionGenerated?.Invoke("[Error: Engine not ready]");
                return;
            }

            string actionPrompt = BuildActionPrompt();

            accumulatedResponse = "";
            isStreaming = true;

            currentStreamId = PythonBridgeService.Instance.StartCompletion(
                actionPrompt,
                onToken: (token) =>
                {
                    accumulatedResponse += token;
                },
                onComplete: () =>
                {
                    OnActionGenerated?.Invoke(accumulatedResponse.Trim());
                    if (debugMode)
                        Debug.Log($"[NeuralAgent] Action: {accumulatedResponse.Trim()}");
                    isStreaming = false;
                },
                onError: (err) =>
                {
                    Debug.LogError($"[NeuralAgent] Action error: {err}");
                    OnActionGenerated?.Invoke($"[Error: {err}]");
                    isStreaming = false;
                },
                temperature: temperature,
                topP: topP,
                maxTokens: maxTokens / 2
            );
        }

        /// <summary>
        /// Build an action prompt based on the agent's state.
        /// </summary>
        private string BuildActionPrompt()
        {
            string nearbyInfo = "No nearby entities detected.";
            if (agentCollider != null && agentCollider.bounds.size.magnitude > 0)
                nearbyInfo = $"Detection radius: {agentCollider.bounds.size.magnitude}m";

            return $"Agent: {agentName}\nGoal: {currentGoal}\nNearby: {nearbyInfo}\nEnvironment: {GetEnvironmentDescription()}\n\nGenerate a single action for {agentName} to perform now:\nAction:";
        }

        private string GetEnvironmentDescription()
        {
            var nearbyObjects = UnityEngine.Object.FindObjectsByType<UnityEngine.GameObject>(FindObjectsInactive.Exclude);
            int count = 0;
            foreach (var obj in nearbyObjects)
            {
                if (obj != gameObject && Vector3.Distance(obj.transform.position, transform.position) < 10f)
                    count++;
            }
            return $"{count} nearby objects";
        }

        /// <summary>
        /// Cancel the current generation if one is in progress.
        /// </summary>
        public void CancelGeneration()
        {
            if (isStreaming && !string.IsNullOrEmpty(currentStreamId))
            {
                PythonBridgeService.Instance?.CancelStream(currentStreamId);
                isStreaming = false;
            }
        }

        /// <summary>
        /// Clear dialogue history.
        /// </summary>
        public void ClearHistory()
        {
            dialogueHistory.Clear();
        }

        /// <summary>
        /// Get history as a readable string.
        /// </summary>
        public string GetHistoryString()
        {
            var sb = new System.Text.StringBuilder();
            foreach (var msg in dialogueHistory)
                sb.AppendLine($"{msg.role}: {msg.content}");
            return sb.ToString();
        }

        /// <summary>
        /// Set the system prompt / personality.
        /// </summary>
        public void SetPersonality(string newPersonality)
        {
            personality = newPersonality;
        }

        void OnDrawGizmosSelected()
        {
            Gizmos.color = IsReady() ? Color.green : Color.red;
            Gizmos.DrawWireSphere(transform.position, 5f);
        }
    }

    /// <summary>
    /// NeuralBehavior: A behavior tree-like system using neural models for decision making.
    /// </summary>
    public class NeuralBehavior : MonoBehaviour
    {
        [Header("Behavior Configuration")]
        public List<BehaviorNode> behaviors = new List<BehaviorNode>();

        public BehaviorNode currentBehavior;

        [Tooltip("Behavior update interval (seconds)")]
        public float updateInterval = 1f;

        [Tooltip("Use neural model for behavior selection")]
        public bool useNeuralSelection = true;

        [Tooltip("Neural agent for behavior generation")]
        public NeuralAgent neuralAgent;

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

        private void UpdateBehavior()
        {
            if (useNeuralSelection && neuralAgent != null && neuralAgent.IsReady())
            {
                string context = BuildBehaviorContext();
                neuralAgent.GenerateDialogue($"Choose a behavior based on: {context}");
            }
            else if (currentBehavior != null)
            {
                currentBehavior.Execute(this);
            }
        }

        private string BuildBehaviorContext()
        {
            string nearby = "nothing nearby";
            if (agentCollider != null)
            {
                var objects = UnityEngine.Object.FindObjectsByType<UnityEngine.GameObject>(FindObjectsInactive.Exclude);
                int count = 0;
                foreach (var obj in objects)
                {
                    if (obj != gameObject && Vector3.Distance(obj.transform.position, transform.position) < 10f)
                        count++;
                }
                nearby = $"{count} objects nearby";
            }

            return $"Player is nearby ({nearby}). Current goal: {(neuralAgent != null ? neuralAgent.currentGoal : "none")}. Time: {System.DateTime.Now:HH:mm}.";
        }

        public void SetBehavior(BehaviorNode behavior)
        {
            currentBehavior = behavior;
        }

        public void AddBehavior(BehaviorNode behavior)
        {
            behaviors.Add(behavior);
        }
    }

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
