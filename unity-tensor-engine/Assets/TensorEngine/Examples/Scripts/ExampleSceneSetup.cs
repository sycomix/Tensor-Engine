using UnityEngine;
using System.Collections.Generic;
using TensorEngine;
using TensorEngine.Bridge;
using TensorEngine.Agents;

namespace TensorEngine.Examples
{
    /// <summary>
    /// Example scene setup script.
    /// Creates a complete demo scene with NPCs, dialogue, and behavior.
    /// </summary>
    public class ExampleSceneSetup : MonoBehaviour
    {
        [Header("Scene Configuration")]
        public string sceneName = "Neural NPC Demo";

        [Header("NPC Configuration")]
        public int npcCount = 3;
        public float npcSpawnRadius = 20f;
        public Vector3 npcSpawnOffset = new Vector3(0, 1, 0);

        [Header("Dialogue Configuration")]
        public string defaultTopic = "Daily life in the village";
        public float dialogueUpdateInterval = 5f;

        [Header("Environment")]
        public string timeOfDay = "Morning";
        public string weather = "Clear";
        public float ambientMood = 0.7f;

        // Internal
        private List<NeuralAgent> spawnedNPCs = new List<NeuralAgent>();
        private NeuralEnvironment neuralEnvironment;
        private MonoBrain monoBrain;
        private PythonBridgeService bridgeService;
        private float nextDialogueUpdate = 0f;

        void Start()
        {
            Debug.Log($"[ExampleSceneSetup] Setting up scene: {sceneName}");

            // Initialize MonoBrain
            monoBrain = FindObjectOfType<MonoBrain>();
            if (monoBrain == null)
            {
                Debug.LogError("[ExampleSceneSetup] No MonoBrain found in scene!");
                return;
            }

            // Initialize Python Bridge
            bridgeService = FindObjectOfType<PythonBridgeService>();
            if (bridgeService == null)
            {
                Debug.LogError("[ExampleSceneSetup] No PythonBridgeService found in scene!");
                return;
            }

            // Initialize Neural Environment
            SetupNeuralEnvironment();

            // Spawn NPCs
            SpawnNPCs();

            // Create dialogue system
            SetupDialogueSystem();

            Debug.Log("[ExampleSceneSetup] Scene setup complete!");
        }

        void Update()
        {
            // Update dialogue system
            if (Time.time >= nextDialogueUpdate)
            {
                UpdateDialogue();
                nextDialogueUpdate = Time.time + dialogueUpdateInterval;
            }

            // Update environment
            if (neuralEnvironment != null)
            {
                neuralEnvironment.timeOfDay = timeOfDay;
                neuralEnvironment.weather = weather;
                neuralEnvironment.ambientMood = ambientMood;
            }
        }

        /// <summary>
        /// Set up the neural environment.
        /// </summary>
        private void SetupNeuralEnvironment()
        {
            var envGO = new GameObject("NeuralEnvironment");
            neuralEnvironment = envGO.AddComponent<NeuralEnvironment>();
            neuralEnvironment.timeOfDay = timeOfDay;
            neuralEnvironment.weather = weather;
            neuralEnvironment.ambientMood = ambientMood;

            // Add some dynamic events
            neuralEnvironment.AddEvent(new NeuralEvent
            {
                name = "Market Day",
                description = "The village market is open today",
                duration = 3600f,
                intensity = 0.8f
            });

            neuralEnvironment.AddEvent(new NeuralEvent
            {
                name = "Festive Lights",
                description = "Decorative lights are hung around the village",
                duration = 7200f,
                intensity = 0.5f
            });

            Debug.Log("[ExampleSceneSetup] Neural environment initialized.");
        }

        /// <summary>
        /// Spawn NPCs in the scene.
        /// </summary>
        private void SpawnNPCs()
        {
            string[] npcNames = { "Village Elder", "Market Merchant", "Forest Guide" };
            string[] personalities = {
                "Wise and kind, speaks in riddles",
                "Bargain-hunter, always looking for a deal",
                "Adventurous, knows the forest well"
            };
            string[] goals = {
                "Greeting visitors",
                "Selling goods",
                "Exploring the forest"
            };

            for (int i = 0; i < npcCount; i++)
            {
                // Create NPC GameObject
                var npcGO = new GameObject($"NPC_{i}");
                npcGO.transform.position = SpawnPosition();

                // Add components
                var agent = npcGO.AddComponent<NeuralAgent>();
                agent.agentName = npcNames[i % npcNames.Length];
                agent.personality = personalities[i % personalities.Length];
                agent.currentGoal = goals[i % goals.Length];
                agent.maxTokens = 150;
                agent.temperature = 0.7f;
                agent.debugMode = true;

                var exampleNPC = npcGO.AddComponent<ExampleNPC>();
                exampleNPC.npcName = agent.agentName;
                exampleNPC.personality = agent.personality;
                exampleNPC.defaultGoal = agent.currentGoal;
                exampleNPC.interactionRange = 5f;

                // Add a simple mesh for visualization
                var meshGO = new GameObject("Mesh");
                meshGO.transform.parent = npcGO.transform;
                meshGO.transform.localPosition = new Vector3(0, 0.5f, 0);
                var meshFilter = meshGO.AddComponent<MeshFilter>();
                var meshRenderer = meshGO.AddComponent<MeshRenderer>();
                var material = new Material(Shader.Find("Standard"));
                material.color = Color.HSVToRGB(i / (float)npcCount, 0.7f, 0.9f);
                meshRenderer.material = material;
                meshFilter.mesh = CreateSimpleCubeMesh();

                spawnedNPCs.Add(agent);

                Debug.Log($"[ExampleSceneSetup] Spawned NPC: {agent.agentName} at {npcGO.transform.position}");
            }

            Debug.Log($"[ExampleSceneSetup] Spawned {npcCount} NPCs.");
        }

        /// <summary>
        /// Set up the dialogue system.
        /// </summary>
        private void SetupDialogueSystem()
        {
            var dialogueGO = new GameObject("DialogueSystem");
            var dialogueSystem = dialogueGO.AddComponent<ExampleProceduralDialogue>();
            dialogueSystem.participants = spawnedNPCs;
            dialogueSystem.conversationTopic = defaultTopic;
            dialogueSystem.setting = "A quiet village square";
            dialogueSystem.updateInterval = 3f;
            dialogueSystem.maxTurns = 10;

            Debug.Log("[ExampleSceneSetup] Dialogue system initialized.");
        }

        /// <summary>
        /// Update NPC dialogue periodically.
        /// </summary>
        private void UpdateDialogue()
        {
            if (spawnedNPCs.Count == 0) return;

            // Select a random NPC to speak
            int npcIdx = Random.Range(0, spawnedNPCs.Count);
            var npc = spawnedNPCs[npcIdx];

            if (!npc.IsReady())
            {
                Debug.LogWarning($"[ExampleSceneSetup] NPC {npc.agentName} model not loaded.");
                return;
            }

            // Generate a random topic
            string[] topics = {
                "The weather today",
                "Village gossip",
                "Local legends",
                "Daily routines",
                "Market prices"
            };
            string topic = topics[Random.Range(0, topics.Length)];

            string prompt = $"Discuss: {topic}";

            npc.GenerateDialogue(prompt, OnNPCSpoke);
        }

        private void OnNPCSpoke(string response)
        {
            Debug.Log($"[ExampleSceneSetup] NPC spoke: {response}");
        }

        /// <summary>
        /// Get a random spawn position within the radius.
        /// </summary>
        private Vector3 SpawnPosition()
        {
            float angle = Random.Range(0f, Mathf.PI * 2f);
            float radius = Random.Range(0f, npcSpawnRadius);
            return new Vector3(
                Mathf.Cos(angle) * radius,
                npcSpawnOffset.y,
                Mathf.Sin(angle) * radius
            ) + npcSpawnOffset;
        }

        /// <summary>
        /// Create a simple cube mesh for visualization.
        /// </summary>
        private Mesh CreateSimpleCubeMesh()
        {
            var mesh = new Mesh();
            mesh.vertices = new Vector3[]
            {
                new Vector3(-0.5f, -0.5f, -0.5f),
                new Vector3(0.5f, -0.5f, -0.5f),
                new Vector3(0.5f, 0.5f, -0.5f),
                new Vector3(-0.5f, 0.5f, -0.5f),
                new Vector3(-0.5f, -0.5f, 0.5f),
                new Vector3(0.5f, -0.5f, 0.5f),
                new Vector3(0.5f, 0.5f, 0.5f),
                new Vector3(-0.5f, 0.5f, 0.5f)
            };
            mesh.uv = new Vector2[]
            {
                new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1), new Vector2(0, 1),
                new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1), new Vector2(0, 1)
            };
            mesh.triangles = new int[]
            {
                0, 1, 2, 0, 2, 3,
                4, 6, 5, 4, 7, 6,
                1, 5, 6, 1, 6, 2,
                0, 3, 7, 0, 7, 4,
                3, 2, 6, 3, 6, 7,
                4, 5, 1, 4, 1, 0
            };
            mesh.RecalculateNormals();
            return mesh;
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(Screen.width - 350, 10, 340, 250));
            GUILayout.Label("[Example Scene Setup]");
            GUILayout.Label($"Scene: {sceneName}");
            GUILayout.Label($"NPCs: {spawnedNPCs.Count}");
            GUILayout.Label($"Time: {timeOfDay}");
            GUILayout.Label($"Weather: {weather}");
            GUILayout.Label($"Mood: {ambientMood:F2}");

            GUILayout.Space(10);
            GUILayout.Label("Environment:");
            GUILayout.Label(neuralEnvironment != null ? neuralEnvironment.GetContextString() : "Not initialized");

            GUILayout.Space(10);
            if (GUILayout.Button("Refresh Dialogue"))
                UpdateDialogue();

            if (GUILayout.Button("Add NPC"))
            {
                SpawnNPCs();
            }

            GUILayout.EndArea();
        }
    }
}
