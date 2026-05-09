using UnityEngine;
using TensorEngine.Agents;

namespace TensorEngine.Examples
{
    /// <summary>
    /// Example: Basic NPC with neural dialogue.
    /// Attach to any NPC GameObject along with a NeuralAgent component.
    /// </summary>
    public class ExampleNPC : MonoBehaviour
    {
        [Header("NPC Configuration")]
        public string npcName = "Village Elder";
        public string personality = "Wise and kind, speaks in riddles";
        public string defaultGoal = "Greeting visitors";

        [Header("Interaction")]
        public float interactionRange = 5f;
        public GameObject interactionUI;
        public bool autoInteract = false;

        [Header("Dialogue Settings")]
        public int maxResponseTokens = 150;
        public float responseTemperature = 0.8f;

        // Internal
        private NeuralAgent neuralAgent;
        private bool isPlayerNearby = false;
        private string lastResponse = "";

        void Start()
        {
            neuralAgent = GetComponent<NeuralAgent>();
            if (neuralAgent == null)
            {
                Debug.LogError("[ExampleNPC] Missing NeuralAgent component!");
                return;
            }

            neuralAgent.agentName = npcName;
            neuralAgent.personality = personality;
            neuralAgent.currentGoal = defaultGoal;
            neuralAgent.maxTokens = maxResponseTokens;
            neuralAgent.temperature = responseTemperature;

            neuralAgent.OnDialogueGenerated += OnDialogueGenerated;
            neuralAgent.OnModelLoaded += OnModelLoaded;
            neuralAgent.OnModelError += OnModelError;

            if (interactionUI != null)
                interactionUI.SetActive(false);
        }

        void Update()
        {
            // Check player proximity
            CheckPlayerProximity();

            // Auto-interact
            if (autoInteract && isPlayerNearby)
            {
                Interact();
            }
        }

        void CheckPlayerProximity()
        {
            var player = GameObject.FindGameObjectWithTag("Player");
            if (player != null)
            {
                float distance = Vector3.Distance(player.transform.position, transform.position);
                isPlayerNearby = distance <= interactionRange;

                if (interactionUI != null)
                    interactionUI.SetActive(isPlayerNearby);
            }
        }

        /// <summary>
        /// Interact with the NPC.
        /// </summary>
        public void Interact()
        {
            if (!isPlayerNearby)
            {
                Debug.Log("[ExampleNPC] Player is too far to interact.");
                return;
            }

            // Generate a greeting
            string greeting = "Hello, traveler. What brings you to our village?";
            neuralAgent.GenerateDialogue(greeting, OnInteractionComplete);
        }

        /// <summary>
        /// Simulate player input for dialogue.
        /// </summary>
        public void RespondToNPC(string playerResponse)
        {
            neuralAgent.GenerateDialogue(playerResponse, OnInteractionComplete);
        }

        private void OnInteractionComplete(string response)
        {
            lastResponse = response;
            Debug.Log($"[ExampleNPC] {npcName}: {response}");

            // Update UI or trigger animation
            if (interactionUI != null)
            {
                // Update dialogue text component if available
                var textComp = interactionUI.GetComponent<UnityEngine.UI.Text>();
                if (textComp != null)
                    textComp.text = response;
            }
        }

        private void OnDialogueGenerated(string response)
        {
            Debug.Log($"[ExampleNPC] Dialogue generated: {response}");
        }

        private void OnModelLoaded()
        {
            Debug.Log($"[ExampleNPC] Model loaded for {npcName}");
        }

        private void OnModelError()
        {
            Debug.LogError($"[ExampleNPC] Model loading failed for {npcName}");
        }

        void OnDrawGizmosSelected()
        {
            // Visualize interaction range
            Gizmos.color = isPlayerNearby ? Color.green : Color.yellow;
            Gizmos.DrawWireSphere(transform.position, interactionRange);
        }

        void OnGUI()
        {
            if (isPlayerNearby)
            {
                GUILayout.BeginArea(new Rect(Screen.width - 320, Screen.height - 200, 300, 180));
                GUILayout.Label($"[NPC: {npcName}]");
                GUILayout.Label($"Personality: {personality}");
                GUILayout.Label($"Goal: {defaultGoal}");
                GUILayout.Label($"Response: {lastResponse}");

                string input = GUILayout.TextArea(lastResponse, GUILayout.Height(80));
                if (GUILayout.Button("Send Response"))
                {
                    RespondToNPC(input);
                }
                GUILayout.EndArea();
            }
        }
    }

    /// <summary>
    /// Example: Procedural dialogue system.
    /// Manages conversations between multiple NPCs.
    /// </summary>
    public class ExampleProceduralDialogue : MonoBehaviour
    {
        [Header("Dialogue Configuration")]
        public float updateInterval = 2f;
        public int maxTurns = 10;
        public float turnTimer = 0f;

        [Header("Participants")]
        public List<NeuralAgent> participants = new List<NeuralAgent>();

        [Header("Context")]
        public string conversationTopic = "Daily life in the village";
        public string setting = "A quiet village square";

        // Internal
        private int currentTurn = 0;
        private List<string> dialogueLog = new List<string>();

        void Start()
        {
            foreach (var agent in participants)
            {
                agent.OnDialogueGenerated += OnAgentSpoke;
            }
        }

        void Update()
        {
            turnTimer += Time.deltaTime;
            if (turnTimer >= updateInterval && currentTurn < maxTurns)
            {
                NextTurn();
                turnTimer = 0f;
            }
        }

        /// <summary>
        /// Start a new conversation.
        /// </summary>
        public void StartConversation()
        {
            currentTurn = 0;
            dialogueLog.Clear();

            if (participants.Count >= 2)
            {
                var topic = $"Discuss: {conversationTopic}";
                participants[0].GenerateDialogue(topic, OnFirstResponse);
            }
        }

        private void OnFirstResponse(string response)
        {
            dialogueLog.Add($"{participants[0].agentName}: {response}");
            NextTurn();
        }

        private void NextTurn()
        {
            if (currentTurn >= participants.Count || currentTurn >= maxTurns)
            {
                Debug.Log("[ExampleProceduralDialogue] Conversation ended.");
                return;
            }

            int agentIdx = currentTurn % participants.Count;
            var agent = participants[agentIdx];

            // Build context from previous dialogue
            string context = BuildDialogueContext();
            string prompt = $"{context}\n\n{agent.agentName}, respond to the conversation about {conversationTopic}:";

            agent.GenerateDialogue(prompt, OnAgentSpoke);
            currentTurn++;
        }

        private string BuildDialogueContext()
        {
            string context = $"Setting: {setting}\n";
            context += $"Topic: {conversationTopic}\n\nPrevious dialogue:\n";
            foreach (var line in dialogueLog)
            {
                context += $"{line}\n";
            }
            return context;
        }

        private void OnAgentSpoke(string response)
        {
            // This is called by the agent's event handler
            // The actual logging happens in the agent itself
        }

        /// <summary>
        /// Get the full dialogue log.
        /// </summary>
        public string GetDialogueLog()
        {
            return string.Join("\n", dialogueLog);
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, Screen.height - 300, 400, 280));
            GUILayout.Label("[Procedural Dialogue]");
            GUILayout.Label($"Turn: {currentTurn}/{maxTurns}");
            GUILayout.Label($"Topic: {conversationTopic}");
            GUILayout.Label($"Setting: {setting}");

            if (GUILayout.Button("Start Conversation"))
                StartConversation();

            GUILayout.Label("Dialogue Log:");
            GUILayout.TextArea(GetDialogueLog(), GUILayout.Height(150));
            GUILayout.EndArea();
        }
    }

    /// <summary>
    /// Example: NPC behavior system with neural decision making.
    /// </summary>
    public class ExampleNeuralBehavior : MonoBehaviour
    {
        [Header("Behavior Configuration")]
        public NeuralAgent neuralAgent;
        public float behaviorUpdateInterval = 3f;
        public float behaviorTimer = 0f;

        [Header("Available Behaviors")]
        public List<string> availableBehaviors = new List<string>
        {
            "Walk around the village",
            "Greet nearby players",
            "Work at the market",
            "Rest at home",
            "Explore the forest"
        };

        [Header("Current State")]
        public string currentBehavior = "Idle";
        public string currentGoal = "Explore the area";

        void Start()
        {
            if (neuralAgent != null)
            {
                neuralAgent.OnActionGenerated += OnActionGenerated;
            }
        }

        void Update()
        {
            behaviorTimer += Time.deltaTime;
            if (behaviorTimer >= behaviorUpdateInterval)
            {
                SelectNewBehavior();
                behaviorTimer = 0f;
            }
        }

        /// <summary>
        /// Use neural model to select a behavior.
        /// </summary>
        private void SelectNewBehavior()
        {
            if (neuralAgent == null || !neuralAgent.IsReady())
            {
                Debug.LogWarning("[ExampleNeuralBehavior] Agent not ready.");
                return;
            }

            string context = $"Current goal: {currentGoal}\nAvailable behaviors: {string.Join(", ", availableBehaviors)}\nTime: {System.DateTime.Now:HH:mm}";
            string prompt = $"Choose a behavior based on: {context}";

            neuralAgent.GenerateDialogue(prompt, OnBehaviorSelected);
        }

        private void OnBehaviorSelected(string response)
        {
            // Parse the response to select a behavior
            foreach (var behavior in availableBehaviors)
            {
                if (response.ToLower().Contains(behavior.ToLower().Substring(0, Mathf.Min(10, behavior.Length))))
                {
                    currentBehavior = behavior;
                    Debug.Log($"[ExampleNeuralBehavior] Selected: {behavior}");
                    ExecuteBehavior(behavior);
                    return;
                }
            }

            // Default behavior
            currentBehavior = availableBehaviors[0];
            Debug.Log($"[ExampleNeuralBehavior] Default: {currentBehavior}");
            ExecuteBehavior(currentBehavior);
        }

        private void ExecuteBehavior(string behavior)
        {
            // Execute the selected behavior (move to location, animate, etc.)
            Debug.Log($"[ExampleNeuralBehavior] Executing: {behavior}");

            // Example: Move to a random location
            var target = new Vector3(
                Random.Range(-50f, 50f),
                transform.position.y,
                Random.Range(-50f, 50f)
            );
            // In a real implementation, you'd use NavMeshAgent to move
        }

        private void OnActionGenerated(string action)
        {
            Debug.Log($"[ExampleNeuralBehavior] Action: {action}");
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(Screen.width - 320, 10, 300, 200));
            GUILayout.Label("[Neural Behavior]");
            GUILayout.Label($"Current: {currentBehavior}");
            GUILayout.Label($"Goal: {currentGoal}");
            GUILayout.Label($"Next update in: {behaviorUpdateInterval - behaviorTimer:F1}s");

            if (GUILayout.Button("Force New Behavior"))
                SelectNewBehavior();

            GUILayout.EndArea();
        }
    }
}
