using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace TensorEngine
{
    [Serializable]
    public class NeuralEvent
    {
        public string name;
        public string description;
        public float duration;
        public float intensity;
        public float elapsed;

        public NeuralEvent()
        {
            name = "";
            description = "";
            duration = 60f;
            intensity = 0.5f;
            elapsed = 0f;
        }

        public float Progress => duration > 0f ? Mathf.Clamp01(elapsed / duration) : 1f;
        public bool IsExpired => elapsed >= duration;

        public void Update(float deltaTime)
        {
            elapsed += deltaTime;
        }
    }

    public class NeuralEnvironment : MonoBehaviour
    {
        [Header("Conditions")]
        public string timeOfDay = "Morning";
        public string weather = "Clear";
        [Range(0f, 1f)]
        public float ambientMood = 0.7f;

        [Header("Events")]
        public List<NeuralEvent> activeEvents = new List<NeuralEvent>();

        public event Action<NeuralEvent> OnEventStarted;
        public event Action<NeuralEvent> OnEventExpired;

        void Update()
        {
            float dt = Time.deltaTime;
            for (int i = activeEvents.Count - 1; i >= 0; i--)
            {
                activeEvents[i].Update(dt);
                if (activeEvents[i].IsExpired)
                {
                    OnEventExpired?.Invoke(activeEvents[i]);
                    activeEvents.RemoveAt(i);
                }
            }
        }

        public void AddEvent(NeuralEvent evt)
        {
            if (evt == null) return;
            activeEvents.Add(evt);
            OnEventStarted?.Invoke(evt);
        }

        public void RemoveEvent(string eventName)
        {
            for (int i = activeEvents.Count - 1; i >= 0; i--)
            {
                if (activeEvents[i].name == eventName)
                {
                    OnEventExpired?.Invoke(activeEvents[i]);
                    activeEvents.RemoveAt(i);
                }
            }
        }

        public void ClearEvents()
        {
            foreach (var evt in activeEvents)
                OnEventExpired?.Invoke(evt);
            activeEvents.Clear();
        }

        public string GetContextString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Time: {timeOfDay}");
            sb.AppendLine($"Weather: {weather}");
            sb.AppendLine($"Mood: {ambientMood:F2}");
            if (activeEvents.Count > 0)
            {
                sb.AppendLine("Active Events:");
                foreach (var evt in activeEvents)
                    sb.AppendLine($"  - {evt.name} ({evt.description}) [{evt.Progress:P0}]");
            }
            return sb.ToString();
        }
    }
}
