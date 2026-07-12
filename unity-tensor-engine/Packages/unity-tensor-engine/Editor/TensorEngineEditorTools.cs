using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using TensorEngine;
using TensorEngine.Agents;
using TensorEngine.Bridge;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace TensorEngine.EditorTools
{
    public sealed class TensorEngineControlPanel : EditorWindow
    {
        private const string EnginePathKey = "TensorEngine.Editor.EnginePath";
        private const string ModelPathKey = "TensorEngine.Editor.ModelPath";
        private const string HostKey = "TensorEngine.Editor.Host";
        private const string PortKey = "TensorEngine.Editor.Port";
        private const string AutoStartKey = "TensorEngine.Editor.AutoStart";

        private string enginePath;
        private string modelPath;
        private string serverHost;
        private int serverPort;
        private bool autoStartEngine;
        private string statusMessage;
        private Vector2 scrollPosition;

        [MenuItem("Tensor Engine/Control Panel", priority = 0)]
        public static void Open()
        {
            var window = GetWindow<TensorEngineControlPanel>("Tensor Engine");
            window.minSize = new Vector2(460f, 420f);
            window.Show();
        }

        [MenuItem("Tensor Engine/Setup/Create Complete Scene Setup", priority = 20)]
        public static void CreateCompleteSceneSetup()
        {
            Undo.IncrementCurrentGroup();
            Undo.SetCurrentGroupName("Create Tensor Engine Scene Setup");
            var group = Undo.GetCurrentGroup();

            var bridge = TensorEngineEditorMenu.EnsureBridgeService();
            var brain = TensorEngineEditorMenu.EnsureMonoBrain();
            TensorEngineEditorMenu.EnsureNeuralEnvironment();

            brain.enginePath = bridge.enginePath;
            brain.modelPath = bridge.modelPath;
            brain.enginePort = bridge.serverPort;
            brain.autoStartEngine = bridge.autoStartEngine;

            EditorUtility.SetDirty(bridge);
            EditorUtility.SetDirty(brain);
            TensorEngineEditorMenu.MarkSceneDirty();
            Undo.CollapseUndoOperations(group);
        }

        private void OnEnable()
        {
            enginePath = EditorPrefs.GetString(EnginePathKey, "engine.exe");
            modelPath = EditorPrefs.GetString(ModelPathKey, "");
            serverHost = EditorPrefs.GetString(HostKey, "127.0.0.1");
            serverPort = EditorPrefs.GetInt(PortKey, 8080);
            autoStartEngine = EditorPrefs.GetBool(AutoStartKey, true);
            statusMessage = "Ready.";
        }

        private void OnGUI()
        {
            scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);
            DrawConfiguration();
            EditorGUILayout.Space(8f);
            DrawSceneTools();
            EditorGUILayout.Space(8f);
            DrawRuntimeTools();
            EditorGUILayout.Space(8f);
            DrawStatus();
            EditorGUILayout.EndScrollView();
        }

        private void DrawConfiguration()
        {
            EditorGUILayout.LabelField("Configuration", EditorStyles.boldLabel);

            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    enginePath = EditorGUILayout.TextField("Engine Path", enginePath);
                    if (GUILayout.Button("Browse", GUILayout.Width(82f)))
                    {
                        var selected = EditorUtility.OpenFilePanel("Select Tensor Engine executable", Directory.GetCurrentDirectory(), "exe");
                        if (!string.IsNullOrEmpty(selected))
                            enginePath = selected;
                    }
                }

                using (new EditorGUILayout.HorizontalScope())
                {
                    modelPath = EditorGUILayout.TextField("Model Path", modelPath);
                    if (GUILayout.Button("Browse", GUILayout.Width(82f)))
                    {
                        var selected = EditorUtility.OpenFolderPanel("Select model directory", Directory.GetCurrentDirectory(), "");
                        if (!string.IsNullOrEmpty(selected))
                            modelPath = selected;
                    }
                }

                serverHost = EditorGUILayout.TextField("Server Host", serverHost);
                serverPort = EditorGUILayout.IntField("Server Port", serverPort);
                if (serverPort < 1 || serverPort > 65535)
                    serverPort = 8080;
                autoStartEngine = EditorGUILayout.Toggle("Auto Start Engine", autoStartEngine);

                if (GUILayout.Button("Save Configuration"))
                    SaveConfiguration();
            }
        }

        private void DrawSceneTools()
        {
            EditorGUILayout.LabelField("Scene Setup", EditorStyles.boldLabel);

            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                if (GUILayout.Button("Create Complete Scene Setup"))
                    CreateCompleteSceneSetup();

                using (new EditorGUILayout.HorizontalScope())
                {
                    if (GUILayout.Button("Create MonoBrain"))
                        TensorEngineEditorMenu.CreateMonoBrain();
                    if (GUILayout.Button("Create Bridge Service"))
                        TensorEngineEditorMenu.CreateBridgeService();
                }

                using (new EditorGUILayout.HorizontalScope())
                {
                    if (GUILayout.Button("Create Environment"))
                        TensorEngineEditorMenu.CreateNeuralEnvironment();
                    if (GUILayout.Button("Add Agent To Selection"))
                        TensorEngineEditorMenu.AddNeuralAgentToSelection();
                }

                if (GUILayout.Button("Apply Configuration To Scene"))
                    ApplyConfigurationToScene();
            }
        }

        private void DrawRuntimeTools()
        {
            EditorGUILayout.LabelField("Runtime", EditorStyles.boldLabel);

            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                using (new EditorGUI.DisabledScope(!Application.isPlaying))
                {
                    using (new EditorGUILayout.HorizontalScope())
                    {
                        if (GUILayout.Button("Start Engine"))
                            StartEngineInPlayMode();
                        if (GUILayout.Button("Stop Engine"))
                            StopEngineInPlayMode();
                    }
                }

                if (GUILayout.Button("Ping /v1/models"))
                    _ = PingEngineAsync();
            }
        }

        private void DrawStatus()
        {
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(statusMessage, MessageType.Info);
        }

        private void SaveConfiguration()
        {
            EditorPrefs.SetString(EnginePathKey, enginePath ?? "");
            EditorPrefs.SetString(ModelPathKey, modelPath ?? "");
            EditorPrefs.SetString(HostKey, string.IsNullOrWhiteSpace(serverHost) ? "127.0.0.1" : serverHost.Trim());
            EditorPrefs.SetInt(PortKey, serverPort);
            EditorPrefs.SetBool(AutoStartKey, autoStartEngine);
            statusMessage = "Configuration saved.";
        }

        private void ApplyConfigurationToScene()
        {
            SaveConfiguration();
            var bridge = TensorEngineEditorMenu.EnsureBridgeService();
            bridge.enginePath = enginePath;
            bridge.modelPath = modelPath;
            bridge.serverHost = string.IsNullOrWhiteSpace(serverHost) ? "127.0.0.1" : serverHost.Trim();
            bridge.serverPort = serverPort;
            bridge.autoStartEngine = autoStartEngine;

            var brain = TensorEngineEditorMenu.EnsureMonoBrain();
            brain.enginePath = enginePath;
            brain.modelPath = modelPath;
            brain.enginePort = serverPort;
            brain.autoStartEngine = autoStartEngine;

            EditorUtility.SetDirty(bridge);
            EditorUtility.SetDirty(brain);
            TensorEngineEditorMenu.MarkSceneDirty();
            statusMessage = "Configuration applied to scene.";
        }

        private void StartEngineInPlayMode()
        {
            var bridge = UnityEngine.Object.FindAnyObjectByType<PythonBridgeService>();
            if (bridge == null)
            {
                statusMessage = "No PythonBridgeService exists in the active scene.";
                return;
            }

            bridge.StartEngine();
            statusMessage = "StartEngine invoked.";
        }

        private void StopEngineInPlayMode()
        {
            var bridge = UnityEngine.Object.FindAnyObjectByType<PythonBridgeService>();
            if (bridge == null)
            {
                statusMessage = "No PythonBridgeService exists in the active scene.";
                return;
            }

            bridge.StopEngine();
            statusMessage = "StopEngine invoked.";
        }

        private async Task PingEngineAsync()
        {
            SaveConfiguration();
            var host = string.IsNullOrWhiteSpace(serverHost) ? "127.0.0.1" : serverHost.Trim();
            var url = $"http://{host}:{serverPort}/v1/models";
            statusMessage = $"Pinging {url}";
            Repaint();

            try
            {
                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
                using var response = await client.GetAsync(url);
                var body = await response.Content.ReadAsStringAsync();
                statusMessage = response.IsSuccessStatusCode
                    ? $"Engine responded: {(int)response.StatusCode} {response.ReasonPhrase}\n{Truncate(body, 700)}"
                    : $"Engine returned: {(int)response.StatusCode} {response.ReasonPhrase}\n{Truncate(body, 700)}";
            }
            catch (Exception ex)
            {
                statusMessage = $"Ping failed: {ex.Message}";
            }

            Repaint();
        }

        private static string Truncate(string value, int maxLength)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
                return value ?? "";
            return value.Substring(0, maxLength) + "\n[truncated]";
        }
    }

    public static class TensorEngineEditorMenu
    {
        [MenuItem("Tensor Engine/Setup/Create MonoBrain", priority = 21)]
        public static void CreateMonoBrain()
        {
            Selection.activeObject = EnsureMonoBrain().gameObject;
        }

        [MenuItem("Tensor Engine/Setup/Create Bridge Service", priority = 22)]
        public static void CreateBridgeService()
        {
            Selection.activeObject = EnsureBridgeService().gameObject;
        }

        [MenuItem("Tensor Engine/Setup/Create Neural Environment", priority = 23)]
        public static void CreateNeuralEnvironment()
        {
            Selection.activeObject = EnsureNeuralEnvironment().gameObject;
        }

        [MenuItem("Tensor Engine/Agents/Add NeuralAgent To Selection", priority = 40)]
        public static void AddNeuralAgentToSelection()
        {
            var target = Selection.activeGameObject;
            if (target == null)
            {
                Debug.LogWarning("[TensorEngine] Select a GameObject before adding a NeuralAgent.");
                return;
            }

            Undo.RegisterFullObjectHierarchyUndo(target, "Add Tensor Engine NeuralAgent");
            if (target.GetComponent<Collider>() == null)
                Undo.AddComponent<CapsuleCollider>(target);
            if (target.GetComponent<NeuralAgent>() == null)
                Undo.AddComponent<NeuralAgent>(target);
            if (target.GetComponent<NeuralBehavior>() == null)
                Undo.AddComponent<NeuralBehavior>(target);
            EditorUtility.SetDirty(target);
            MarkSceneDirty();
        }

        [MenuItem("Tensor Engine/Agents/Add NeuralAgent To Selection", true)]
        public static bool ValidateAddNeuralAgentToSelection()
        {
            return Selection.activeGameObject != null;
        }

        [MenuItem("Tensor Engine/Tools/Open Persistent Data Path", priority = 70)]
        public static void OpenPersistentDataPath()
        {
            if (!Directory.Exists(Application.persistentDataPath))
                Directory.CreateDirectory(Application.persistentDataPath);
            EditorUtility.RevealInFinder(Application.persistentDataPath);
        }

        [MenuItem("Tensor Engine/Tools/Reimport Tensor Engine Assets", priority = 71)]
        public static void ReimportTensorEngineAssets()
        {
            AssetDatabase.Refresh(ImportAssetOptions.ForceUpdate);
        }

        internal static MonoBrain EnsureMonoBrain()
        {
            var existing = UnityEngine.Object.FindAnyObjectByType<MonoBrain>();
            if (existing != null)
                return existing;

            var go = new GameObject("MonoBrain");
            Undo.RegisterCreatedObjectUndo(go, "Create Tensor Engine MonoBrain");
            var brain = go.AddComponent<MonoBrain>();
            MarkSceneDirty();
            return brain;
        }

        internal static PythonBridgeService EnsureBridgeService()
        {
            var existing = UnityEngine.Object.FindAnyObjectByType<PythonBridgeService>();
            if (existing != null)
                return existing;

            var go = new GameObject("PythonBridgeService");
            Undo.RegisterCreatedObjectUndo(go, "Create Tensor Engine Bridge Service");
            var bridge = go.AddComponent<PythonBridgeService>();
            MarkSceneDirty();
            return bridge;
        }

        internal static NeuralEnvironment EnsureNeuralEnvironment()
        {
            var existing = UnityEngine.Object.FindAnyObjectByType<NeuralEnvironment>();
            if (existing != null)
                return existing;

            var go = new GameObject("NeuralEnvironment");
            Undo.RegisterCreatedObjectUndo(go, "Create Tensor Engine Neural Environment");
            var environment = go.AddComponent<NeuralEnvironment>();
            MarkSceneDirty();
            return environment;
        }

        internal static void MarkSceneDirty()
        {
            if (!Application.isPlaying)
                EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }
    }

    [CustomEditor(typeof(MonoBrain))]
    public sealed class MonoBrainInspector : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var brain = (MonoBrain)target;
            EditorGUILayout.Space();
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                if (GUILayout.Button("Open Tensor Engine Control Panel"))
                    TensorEngineControlPanel.Open();

                using (new EditorGUI.DisabledScope(!Application.isPlaying))
                {
                    if (GUILayout.Button("Generate Dialogue For Registered Agents"))
                    {
                        foreach (var agent in brain.registeredAgents)
                        {
                            if (agent != null)
                                brain.GenerateAgentDialogue(agent, "Introduce yourself and describe your current goal.");
                        }
                    }
                }
            }
        }
    }

    [CustomEditor(typeof(PythonBridgeService))]
    public sealed class PythonBridgeServiceInspector : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var bridge = (PythonBridgeService)target;
            EditorGUILayout.Space();
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("Server URL", bridge.serverUrl);
                using (new EditorGUI.DisabledScope(!Application.isPlaying))
                {
                    using (new EditorGUILayout.HorizontalScope())
                    {
                        if (GUILayout.Button("Start Engine"))
                            bridge.StartEngine();
                        if (GUILayout.Button("Stop Engine"))
                            bridge.StopEngine();
                    }
                }

                if (GUILayout.Button("Open Tensor Engine Control Panel"))
                    TensorEngineControlPanel.Open();
            }
        }
    }
}
