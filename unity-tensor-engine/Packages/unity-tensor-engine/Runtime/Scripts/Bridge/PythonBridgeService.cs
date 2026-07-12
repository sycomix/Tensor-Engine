using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Newtonsoft.Json;
using UnityEngine.Networking;

namespace TensorEngine.Bridge
{
    public class PythonBridgeService : MonoBehaviour
    {
        public static PythonBridgeService Instance { get; private set; }

        [Header("Engine Settings")]
        [Tooltip("Path to engine.exe")]
        public string enginePath = "engine.exe";

        [Tooltip("Path to the model directory")]
        public string modelPath = "";

        [Tooltip("Path to tokenizer file (optional, auto-derived from modelPath)")]
        public string tokenizerPath = "";

        [Tooltip("Path to the config directory (optional)")]
        public string configPath = "";

        [Tooltip("Port for the engine HTTP server (default 8080)")]
        public int serverPort = 8080;

        [Tooltip("Auto-start the engine on Awake")]
        public bool autoStartEngine = true;

        [Tooltip("Engine host address")]
        public string serverHost = "127.0.0.1";

        public string serverUrl => $"http://{serverHost}:{serverPort}";

        public bool IsRunning { get; private set; }

        [Header("Advanced")]
        [Tooltip("Max sequence length for generation")]
        public int maxSeqLen = 2048;

        [Tooltip("Prompt cache size")]
        public int promptCacheSize = 128;

        [Tooltip("Additional CLI arguments for engine.exe")]
        public string extraArgs = "";

        private readonly ConcurrentQueue<SSEEvent> sseQueue = new ConcurrentQueue<SSEEvent>();
        private readonly HttpClient httpClient = new HttpClient { Timeout = TimeSpan.FromMilliseconds(-1) };
        private readonly Dictionary<string, CancellationTokenSource> streamCtsMap = new Dictionary<string, CancellationTokenSource>();
        private readonly Dictionary<string, List<Action<string>>> streamCallbacks = new Dictionary<string, List<Action<string>>>();
        private readonly Dictionary<string, Action> streamCompleteCallbacks = new Dictionary<string, Action>();
        private readonly Dictionary<string, Action<string>> streamErrorCallbacks = new Dictionary<string, Action<string>>();

        private System.Diagnostics.Process engineProcess;
        private string modelId;

        public struct SSEEvent
        {
            public string streamId;
            public string token;
            public bool isDone;
            public string error;
        }

        void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);
            httpClient.DefaultRequestHeaders.Add("Accept", "text/event-stream");
        }

        void Start()
        {
            if (autoStartEngine)
                StartEngine();
        }

        void Update()
        {
            while (sseQueue.TryDequeue(out SSEEvent evt))
            {
                if (!string.IsNullOrEmpty(evt.error))
                {
                    if (streamErrorCallbacks.TryGetValue(evt.streamId, out var errCb))
                    {
                        errCb?.Invoke(evt.error);
                        streamErrorCallbacks.Remove(evt.streamId);
                    }
                    streamCallbacks.Remove(evt.streamId);
                    streamCompleteCallbacks.Remove(evt.streamId);
                    streamCtsMap.Remove(evt.streamId);
                    continue;
                }

                if (evt.isDone)
                {
                    if (streamCompleteCallbacks.TryGetValue(evt.streamId, out var doneCb))
                    {
                        doneCb?.Invoke();
                        streamCompleteCallbacks.Remove(evt.streamId);
                    }
                    streamCallbacks.Remove(evt.streamId);
                    streamErrorCallbacks.Remove(evt.streamId);
                    streamCtsMap.Remove(evt.streamId);
                    continue;
                }

                if (streamCallbacks.TryGetValue(evt.streamId, out var cbs))
                {
                    foreach (var cb in cbs)
                        cb?.Invoke(evt.token);
                }
            }
        }

        void OnDestroy()
        {
            StopEngine();
            httpClient.Dispose();
        }

        public void StartEngine()
        {
            if (IsRunning)
            {
                Debug.LogWarning("[TensorEngine] Engine already running.");
                return;
            }

            if (string.IsNullOrEmpty(modelPath) || !Directory.Exists(modelPath))
            {
                Debug.LogError($"[TensorEngine] Model path not found: {modelPath}");
                return;
            }

            modelId = new DirectoryInfo(modelPath).Name;

            if (string.IsNullOrEmpty(tokenizerPath))
            {
                string possible = Path.Combine(modelPath, "tokenizer.model");
                if (File.Exists(possible))
                    tokenizerPath = possible;
                else
                {
                    possible = Path.Combine(modelPath, "tokenizer.json");
                    if (File.Exists(possible))
                        tokenizerPath = possible;
                }
            }

            string args = $"--model-path \"{modelPath}\"";
            if (!string.IsNullOrEmpty(tokenizerPath))
                args += $" --tokenizer-path \"{tokenizerPath}\"";
            if (!string.IsNullOrEmpty(configPath))
                args += $" --param-path \"{configPath}\"";
            args += $" --max-seq-len {maxSeqLen}";
            args += $" --inference-server-port {serverPort}";
            args += $" --inference-server-host {serverHost}";
            args += $" --inference-server-prompt-cache-size {promptCacheSize}";
            args += $" --quiet";
            if (!string.IsNullOrEmpty(extraArgs))
                args += " " + extraArgs;

            try
            {
                engineProcess = new System.Diagnostics.Process();
                engineProcess.StartInfo.FileName = enginePath;
                engineProcess.StartInfo.Arguments = args;
                engineProcess.StartInfo.UseShellExecute = false;
                engineProcess.StartInfo.RedirectStandardOutput = true;
                engineProcess.StartInfo.RedirectStandardError = true;
                engineProcess.StartInfo.CreateNoWindow = true;
                engineProcess.OutputDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Debug.Log($"[Engine] {e.Data}");
                };
                engineProcess.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Debug.LogError($"[Engine] {e.Data}");
                };
                engineProcess.Start();
                engineProcess.BeginOutputReadLine();
                engineProcess.BeginErrorReadLine();

                Debug.Log($"[TensorEngine] Starting engine on {serverUrl}...");
                IsRunning = true;

                StartCoroutine(WaitForServerReady());
            }
            catch (Exception e)
            {
                Debug.LogError($"[TensorEngine] Failed to start engine: {e.Message}");
                IsRunning = false;
            }
        }

        private System.Collections.IEnumerator WaitForServerReady()
        {
            for (int i = 0; i < 60; i++)
            {
                var www = UnityWebRequest.Get($"{serverUrl}/v1/models");
                www.timeout = 2;
                yield return www.SendWebRequest();

                if (www.result != UnityWebRequest.Result.ConnectionError && www.responseCode == 200)
                {
                    Debug.Log("[TensorEngine] Engine is ready!");
                    yield break;
                }
                yield return new WaitForSeconds(1);
            }
            Debug.LogWarning("[TensorEngine] Engine did not become ready within 60s.");
        }

        public void StopEngine()
        {
            if (!IsRunning) return;

            foreach (var kvp in streamCtsMap)
                kvp.Value?.Cancel();
            streamCtsMap.Clear();

            try
            {
                if (engineProcess != null && !engineProcess.HasExited)
                {
                    engineProcess.Kill();
                    engineProcess.WaitForExit(5000);
                }
            }
            catch { }

            IsRunning = false;
            engineProcess = null;
            streamCallbacks.Clear();
            streamCompleteCallbacks.Clear();
            streamErrorCallbacks.Clear();
            Debug.Log("[TensorEngine] Engine stopped.");
        }

        public async Task<string[]> ListModelsAsync()
        {
            var www = UnityWebRequest.Get($"{serverUrl}/v1/models");
            www.timeout = 10;
            await Task.Yield();
            _ = www.SendWebRequest();

            while (!www.isDone)
                await Task.Yield();

            if (www.result == UnityWebRequest.Result.ConnectionError)
            {
                Debug.LogError($"[TensorEngine] Failed to list models: {www.error}");
                return Array.Empty<string>();
            }

            try
            {
                var response = JsonConvert.DeserializeObject<ModelsResponse>(www.downloadHandler.text);
                var ids = new List<string>();
                foreach (var entry in response.data)
                    ids.Add(entry.id);
                return ids.ToArray();
            }
            catch (Exception e)
            {
                Debug.LogError($"[TensorEngine] Failed to parse models: {e.Message}");
                return Array.Empty<string>();
            }
        }

        public string GetCurrentModelId() => modelId;

        public string StartChatCompletion(
            List<ChatMessage> messages,
            Action<string> onToken,
            Action onComplete,
            Action<string> onError = null,
            float temperature = 0.7f,
            float topP = 0.95f,
            int maxTokens = 256,
            int topK = 40,
            float repetitionPenalty = 1.1f)
        {
            string streamId = Guid.NewGuid().ToString();

            streamCallbacks[streamId] = new List<Action<string>> { onToken };
            streamCompleteCallbacks[streamId] = onComplete;
            if (onError != null)
                streamErrorCallbacks[streamId] = onError;

            var payload = new Dictionary<string, object>
            {
                ["model"] = modelId,
                ["messages"] = messages.ConvertAll(m => new Dictionary<string, string>
                {
                    ["role"] = m.role,
                    ["content"] = m.content
                }),
                ["max_tokens"] = maxTokens,
                ["temperature"] = temperature,
                ["top_p"] = topP,
                ["top_k"] = topK,
                ["repetition_penalty"] = repetitionPenalty,
                ["stream"] = true
            };

            string json = JsonConvert.SerializeObject(payload);
            RunSSEStream(streamId, $"{serverUrl}/v1/chat/completions", json);
            return streamId;
        }

        public string StartCompletion(
            string prompt,
            Action<string> onToken,
            Action onComplete,
            Action<string> onError = null,
            float temperature = 0.7f,
            float topP = 0.95f,
            int maxTokens = 256,
            int topK = 40,
            float repetitionPenalty = 1.1f)
        {
            string streamId = Guid.NewGuid().ToString();

            streamCallbacks[streamId] = new List<Action<string>> { onToken };
            streamCompleteCallbacks[streamId] = onComplete;
            if (onError != null)
                streamErrorCallbacks[streamId] = onError;

            var payload = new Dictionary<string, object>
            {
                ["model"] = modelId,
                ["prompt"] = prompt,
                ["max_tokens"] = maxTokens,
                ["temperature"] = temperature,
                ["top_p"] = topP,
                ["top_k"] = topK,
                ["repetition_penalty"] = repetitionPenalty,
                ["stream"] = true
            };

            string json = JsonConvert.SerializeObject(payload);
            RunSSEStream(streamId, $"{serverUrl}/v1/completions", json);
            return streamId;
        }

        public void CancelStream(string streamId)
        {
            if (streamCtsMap.TryGetValue(streamId, out var cts))
            {
                cts.Cancel();
                streamCtsMap.Remove(streamId);
            }
            streamCallbacks.Remove(streamId);
            streamCompleteCallbacks.Remove(streamId);
            streamErrorCallbacks.Remove(streamId);
        }

        public async Task<string> ChatCompletionAsync(
            List<ChatMessage> messages,
            float temperature = 0.7f,
            float topP = 0.95f,
            int maxTokens = 256,
            int topK = 40,
            float repetitionPenalty = 1.1f)
        {
            var payload = new Dictionary<string, object>
            {
                ["model"] = modelId,
                ["messages"] = messages.ConvertAll(m => new Dictionary<string, string>
                {
                    ["role"] = m.role,
                    ["content"] = m.content
                }),
                ["max_tokens"] = maxTokens,
                ["temperature"] = temperature,
                ["top_p"] = topP,
                ["top_k"] = topK,
                ["repetition_penalty"] = repetitionPenalty,
                ["stream"] = false
            };

            string json = JsonConvert.SerializeObject(payload);
            var www = new UnityWebRequest($"{serverUrl}/v1/chat/completions", "POST");
            byte[] body = Encoding.UTF8.GetBytes(json);
            www.uploadHandler = new UploadHandlerRaw(body);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = 120;

            await Task.Yield();
            _ = www.SendWebRequest();

            while (!www.isDone)
                await Task.Yield();

            if (www.result == UnityWebRequest.Result.ConnectionError || www.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError($"[TensorEngine] Chat completion error: {www.error}");
                return null;
            }

            return ParseChatResponse(www.downloadHandler.text);
        }

        private void RunSSEStream(string streamId, string url, string json)
        {
            var cts = new CancellationTokenSource();
            streamCtsMap[streamId] = cts;
            var ct = cts.Token;

            Task.Run(async () =>
            {
                try
                {
                    using var content = new StringContent(json, Encoding.UTF8, "application/json");
                    using var response = await httpClient.PostAsync(url, content, ct);
                    response.EnsureSuccessStatusCode();

                    using var stream = await response.Content.ReadAsStreamAsync();
                    using var reader = new StreamReader(stream);

                    while (!reader.EndOfStream && !ct.IsCancellationRequested)
                    {
                        string line = await reader.ReadLineAsync();
                        if (line == null) break;

                        if (line.StartsWith("data: "))
                        {
                            string data = line.Substring(6);
                            if (data == "[DONE]")
                            {
                                sseQueue.Enqueue(new SSEEvent { streamId = streamId, isDone = true });
                                return;
                            }
                            string token = ParseSSEToken(data);
                            if (token != null)
                            {
                                sseQueue.Enqueue(new SSEEvent { streamId = streamId, token = token });
                            }
                        }
                    }
                    sseQueue.Enqueue(new SSEEvent { streamId = streamId, isDone = true });
                }
                catch (OperationCanceledException) { }
                catch (Exception e)
                {
                    sseQueue.Enqueue(new SSEEvent { streamId = streamId, error = e.Message });
                }
            }, ct);
        }

        private string ParseSSEToken(string data)
        {
            try
            {
                var chunk = JsonConvert.DeserializeObject<SSEChunk>(data);
                if (chunk?.choices != null && chunk.choices.Count > 0)
                {
                    var choice = chunk.choices[0];
                    if (choice.delta != null && !string.IsNullOrEmpty(choice.delta.content))
                        return choice.delta.content;
                    if (!string.IsNullOrEmpty(choice.text))
                        return choice.text;
                }
            }
            catch { }
            return null;
        }

        private string ParseChatResponse(string json)
        {
            try
            {
                var response = JsonConvert.DeserializeObject<ChatResponse>(json);
                if (response?.choices != null && response.choices.Count > 0)
                    return response.choices[0].message?.content;
            }
            catch { }
            return null;
        }

        [Serializable]
        public class ModelsResponse
        {
            public string @object;
            public List<ModelEntry> data;
        }

        [Serializable]
        public class ModelEntry
        {
            public string id;
            public string @object;
        }

        [Serializable]
        public class SSEChunk
        {
            public List<SSEChoice> choices;
        }

        [Serializable]
        public class SSEChoice
        {
            public SSEDelta delta;
            public string text;
            public string finish_reason;
        }

        [Serializable]
        public class SSEDelta
        {
            public string role;
            public string content;
        }

        [Serializable]
        public class ChatResponse
        {
            public List<ChatResponseChoice> choices;
        }

        [Serializable]
        public class ChatResponseChoice
        {
            public ChatResponseMessage message;
        }

        [Serializable]
        public class ChatResponseMessage
        {
            public string role;
            public string content;
        }
    }

    [Serializable]
    public class ChatMessage
    {
        public string role;
        public string content;

        public ChatMessage(string role, string content)
        {
            this.role = role;
            this.content = content;
        }
    }
}
