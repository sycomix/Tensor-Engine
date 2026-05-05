using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.Networking;
using TensorEngine.Core;

namespace TensorEngine.Bridge
{
    /// <summary>
    /// Python bridge service. Manages a Python subprocess running the Tensor-Engine inference server.
    /// Communicates via HTTP REST API for model operations and tensor computation.
    /// </summary>
    public class PythonBridgeService : MonoBehaviour
    {
        public static PythonBridgeService Instance { get; private set; }

        [Header("Bridge Settings")]
        [Tooltip("Path to the Python interpreter")]
        public string pythonPath = "python3";

        [Tooltip("Path to the Tensor-Engine Python bridge script")]
        public string bridgeScriptPath = "";

        [Tooltip("Port for the local inference server")]
        public int serverPort = 8765;

        [Tooltip("Auto-start the server on Awake")]
        public bool autoStartServer = true;

        [Tooltip("Server base URL")]
        public string serverUrl => $"http://localhost:{serverPort}";

        [Tooltip("Request timeout in seconds")]
        public int requestTimeout = 30;

        public bool IsRunning { get; private set; }
        private Process serverProcess;
        private string serverOutputLog;

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
            if (autoStartServer)
                StartServer();
        }

        void OnDestroy()
        {
            StopServer();
        }

        /// <summary>
        /// Start the Tensor-Engine inference server as a subprocess.
        /// </summary>
        public void StartServer()
        {
            if (IsRunning)
            {
                Debug.LogWarning("[TensorEngine] Server already running.");
                return;
            }

            if (string.IsNullOrEmpty(bridgeScriptPath))
            {
                // Default: use the PythonBridge directory
                bridgeScriptPath = Path.Combine(
                    Directory.GetCurrentDirectory(),
                    "Assets", "TensorEngine", "PythonBridge", "services", "inference_server.py"
                );
            }

            if (!File.Exists(bridgeScriptPath))
            {
                Debug.LogError($"[TensorEngine] Bridge script not found: {bridgeScriptPath}");
                Debug.LogWarning("[TensorEngine] You need to set up the Python bridge first. Run 'python3 setup.py install' in the PythonBridge directory.");
                return;
            }

            string args = $"-u {bridgeScriptPath} --port {serverPort} --host 0.0.0.0";

            try
            {
                serverProcess = new Process();
                serverProcess.StartInfo.FileName = pythonPath;
                serverProcess.StartInfo.Arguments = args;
                serverProcess.StartInfo.UseShellExecute = false;
                serverProcess.StartInfo.RedirectStandardOutput = true;
                serverProcess.StartInfo.RedirectStandardError = true;
                serverProcess.StartInfo.CreateNoWindow = true;
                serverProcess.OutputDataReceived += OnOutputData;
                serverProcess.ErrorDataReceived += OnErrorData;
                serverProcess.Start();
                serverProcess.BeginOutputReadLine();
                serverProcess.BeginErrorReadLine();

                Debug.Log($"[TensorEngine] Starting inference server on port {serverPort}...");
                IsRunning = true;

                // Wait for server to be ready
                WaitForServerReady();
            }
            catch (Exception e)
            {
                Debug.LogError($"[TensorEngine] Failed to start server: {e.Message}");
                IsRunning = false;
            }
        }

        private void WaitForServerReady()
        {
            for (int i = 0; i < 30; i++)
            {
                try
                {
                    var www = UnityWebRequest.Get($"{serverUrl}/health");
                    www.timeout = 2;
                    www.SendWebRequest();
                    if (!www.isNetworkError && www.responseCode == 200)
                    {
                        Debug.Log("[TensorEngine] Server is ready!");
                        return;
                    }
                }
                catch { /* not ready yet */ }
                Thread.Sleep(1000);
            }
            Debug.LogWarning("[TensorEngine] Server may not be ready yet. Continuing anyway...");
        }

        /// <summary>
        /// Stop the inference server.
        /// </summary>
        public void StopServer()
        {
            if (!IsRunning) return;

            try
            {
                if (serverProcess != null && !serverProcess.HasExited)
                {
                    serverProcess.Kill();
                    serverProcess.WaitForExit();
                }
            }
            catch { }

            IsRunning = false;
            serverProcess = null;
            Debug.Log("[TensorEngine] Inference server stopped.");
        }

        private void OnOutputData(object sender, DataReceivedEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                serverOutputLog = e.Data;
                Debug.Log($"[TE Server] {e.Data}");
            }
        }

        private void OnErrorData(object sender, DataReceivedEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                Debug.LogError($"[TE Server] {e.Data}");
            }
        }

        /// <summary>
        /// Send a tensor computation request to the Python server.
        /// Returns the result as a Tensor.
        /// </summary>
        public async System.Threading.Tasks.Task<Tensor> ComputeAsync(string operation, Tensor input, Tensor[] inputs = null, float[] floatArgs = null)
        {
            var payload = new System.Collections.Generic.Dictionary<string, object>
            {
                { "operation", operation },
                { "input", input != null ? input.ToJson() : null },
                { "inputs", inputs != null ? System.Text.Json.JsonSerializer.Serialize(inputs) : null },
                { "float_args", floatArgs != null ? System.Text.Json.JsonSerializer.Serialize(floatArgs) : null }
            };

            string json = System.Text.Json.JsonSerializer.Serialize(payload);
            var www = new UnityWebRequest(serverUrl + "/compute", "POST");
            var body = System.Text.Encoding.UTF8.GetBytes(json);
            www.uploadHandler = new UploadHandlerRaw(body);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = requestTimeout;

            await System.Threading.Tasks.Task.Yield();
            www.SendWebRequest();

            while (!www.isDone)
                await System.Threading.Tasks.Task.Yield();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.LogError($"[TensorEngine] Compute error: {www.error}");
                return null;
            }

            return Tensor.FromJson(www.downloadHandler.text);
        }

        /// <summary>
        /// Load a model from a SafeTensors file on the server.
        /// </summary>
        public async System.Threading.Tasks.Task<bool> LoadModelAsync(string modelId, string modelPath)
        {
            var payload = new System.Collections.Generic.Dictionary<string, object>
            {
                { "model_id", modelId },
                { "model_path", modelPath }
            };
            string json = System.Text.Json.JsonSerializer.Serialize(payload);
            var www = new UnityWebRequest(serverUrl + "/models/load", "POST");
            var body = System.Text.Encoding.UTF8.GetBytes(json);
            www.uploadHandler = new UploadHandlerRaw(body);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = 60;

            await System.Threading.Tasks.Task.Yield();
            www.SendWebRequest();

            while (!www.isDone)
                await System.Threading.Tasks.Task.Yield();

            bool success = !www.isNetworkError && www.responseCode == 200;
            if (!success)
                Debug.LogError($"[TensorEngine] Load model error: {www.error}");
            return success;
        }

        /// <summary>
        /// Run inference with a loaded model.
        /// </summary>
        public async System.Threading.Tasks.Task<Tensor> InferenceAsync(string modelId, Tensor inputIds, int maxTokens = 100, float temperature = 0.8f)
        {
            var payload = new System.Collections.Generic.Dictionary<string, object>
            {
                { "model_id", modelId },
                { "input", inputIds.ToJson() },
                { "max_tokens", maxTokens },
                { "temperature", temperature }
            };
            string json = System.Text.Json.JsonSerializer.Serialize(payload);
            var www = new UnityWebRequest(serverUrl + "/inference", "POST");
            var body = System.Text.Encoding.UTF8.GetBytes(json);
            www.uploadHandler = new UploadHandlerRaw(body);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = requestTimeout;

            await System.Threading.Tasks.Task.Yield();
            www.SendWebRequest();

            while (!www.isDone)
                await System.Threading.Tasks.Task.Yield();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.LogError($"[TensorEngine] Inference error: {www.error}");
                return null;
            }

            return Tensor.FromJson(www.downloadHandler.text);
        }

        /// <summary>
        /// List loaded models.
        /// </summary>
        public async System.Threading.Tasks.Task<string[]> ListModelsAsync()
        {
            var www = UnityWebRequest.Get(serverUrl + "/models");
            www.timeout = 10;
            await System.Threading.Tasks.Task.Yield();
            www.SendWebRequest();

            while (!www.isDone)
                await System.Threading.Tasks.Task.Yield();

            if (www.isNetworkError) return new string[0];

            var obj = System.Text.Json.JsonSerializer.Deserialize<System.Collections.Generic.Dictionary<string, object>>(www.downloadHandler.text);
            if (obj.ContainsKey("models"))
                return System.Text.Json.JsonSerializer.Deserialize<string[]>(obj["models"].ToString());
            return new string[0];
        }
    }
}
