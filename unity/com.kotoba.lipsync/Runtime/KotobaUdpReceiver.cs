using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace Kotoba.LipSync
{
    /// <summary>External CPU inference receiver for a normal Unity application.
    /// VRChat avatars use the OSC/FX builder; this MonoBehaviour is not uploaded.</summary>
    public sealed class KotobaUdpReceiver : MonoBehaviour
    {
        public enum ShapeMode { Oculus15, FiveVowels }
        public SkinnedMeshRenderer face;
        public int port = 39541;
        public bool allowRemoteHost = false;
        public ShapeMode mode = ShapeMode.Oculus15;
        public string[] visemeShapes = {
            "vrc.v_sil", "vrc.v_pp", "vrc.v_ff", "vrc.v_th", "vrc.v_dd",
            "vrc.v_kk", "vrc.v_ch", "vrc.v_ss", "vrc.v_nn", "vrc.v_rr",
            "vrc.v_aa", "vrc.v_e", "vrc.v_ih", "vrc.v_oh", "vrc.v_ou"
        };
        public string[] vowelShapes = { "A", "I", "U", "E", "O" };
        [Tooltip("Optional blendshape for a gentle breath opening; leave empty if absent.")]
        public string breathShape = "";
        [Range(0.05f, 2f)] public float staleAfterSeconds = 0.3f;
        [Range(0f, 100f)] public float gain = 100f;

        [Serializable]
        public sealed class Packet
        {
            public string protocol;
            public float timestamp;
            public string viseme;
            public int viseme_id;
            public float confidence;
            public float[] weights;
            public float[] vrm;
            public float breath;
            public float nasal;
            public float rms_db;
            public float[] formants_hz;
            public float inference_ms;
            public int sequence;
        }

        public Packet Latest { get; private set; }
        private UdpClient client;
        private Thread thread;
        private volatile bool running;
        private readonly object gate = new object();
        private string pending;
        private int[] indices;
        private int breathIndex = -1;
        private float lastReceived = -100;

        private void OnEnable()
        {
            if (!face || !face.sharedMesh)
            {
                Debug.LogError("Kotoba: assign a face SkinnedMeshRenderer.", this);
                enabled = false;
                return;
            }
            string[] shapes = mode == ShapeMode.Oculus15 ? visemeShapes : vowelShapes;
            int expected = mode == ShapeMode.Oculus15 ? 15 : 5;
            if (shapes == null || shapes.Length != expected || port < 1 || port > 65535)
            {
                Debug.LogError("Kotoba: invalid blendshape array or port.", this);
                enabled = false;
                return;
            }
            indices = new int[shapes.Length];
            for (int i = 0; i < indices.Length; i++)
                indices[i] = face.sharedMesh.GetBlendShapeIndex(shapes[i]);
            breathIndex = string.IsNullOrEmpty(breathShape) ? -1 : face.sharedMesh.GetBlendShapeIndex(breathShape);
            try
            {
                client = new UdpClient(new IPEndPoint(allowRemoteHost ? IPAddress.Any : IPAddress.Loopback, port));
                running = true;
                lastReceived = -100;
                thread = new Thread(Receive) { IsBackground = true, Name = "Kotoba LipSync UDP" };
                thread.Start();
            }
            catch (SocketException e)
            {
                Debug.LogError("Kotoba: cannot open UDP: " + e.Message, this);
                enabled = false;
            }
        }

        private void Receive()
        {
            var endpoint = new IPEndPoint(IPAddress.Any, 0);
            while (running)
            {
                try
                {
                    byte[] bytes = client.Receive(ref endpoint);
                    if (bytes.Length > 8192) continue;
                    string text = Encoding.UTF8.GetString(bytes);
                    lock (gate) pending = text; // Keep latest only. No Unity API off main thread.
                }
                catch (ObjectDisposedException) { break; }
                catch (SocketException) { if (!running) break; }
            }
        }

        private static bool Valid(float[] values, int count)
        {
            if (values == null || values.Length != count) return false;
            foreach (float value in values)
                if (float.IsNaN(value) || float.IsInfinity(value) || value < 0 || value > 1.001f) return false;
            return true;
        }

        private void Update()
        {
            string text;
            lock (gate) { text = pending; pending = null; }
            if (text != null)
            {
                try
                {
                    Packet value = JsonUtility.FromJson<Packet>(text);
                    if (value != null && value.protocol == "kotoba-lipsync/1" && Valid(value.weights, 15) && Valid(value.vrm, 5)
                        && !float.IsNaN(value.breath) && !float.IsInfinity(value.breath))
                    {
                        Latest = value;
                        lastReceived = Time.unscaledTime;
                    }
                }
                catch (ArgumentException) { /* Ignore malformed UDP. */ }
            }
            bool fresh = Latest != null && Time.unscaledTime - lastReceived < staleAfterSeconds;
            float[] values = fresh ? (mode == ShapeMode.Oculus15 ? Latest.weights : Latest.vrm) : null;
            if (!face) return;
            for (int i = 0; i < indices.Length; i++)
                if (indices[i] >= 0) face.SetBlendShapeWeight(indices[i], values == null ? 0 : Mathf.Clamp01(values[i]) * gain);
            if (breathIndex >= 0) face.SetBlendShapeWeight(breathIndex, fresh ? Mathf.Clamp01(Latest.breath) * gain : 0);
        }

        private void OnDisable()
        {
            running = false;
            client?.Close();
            thread?.Join(500);
            thread = null;
            client = null;
            lock (gate) pending = null;
            Latest = null;
            if (face && indices != null)
            {
                foreach (int index in indices) if (index >= 0) face.SetBlendShapeWeight(index, 0);
                if (breathIndex >= 0) face.SetBlendShapeWeight(breathIndex, 0);
            }
        }
    }
}
