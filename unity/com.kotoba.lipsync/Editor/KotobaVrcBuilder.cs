// Reflection isolates this optional tool from VRChat SDK assembly versions.
// It builds actual AnimationClips/BlendTree assets; no runtime script is uploaded.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace Kotoba.LipSync.Editor
{
    public sealed class KotobaVrcBuilder : EditorWindow
    {
        private GameObject avatar;
        private SkinnedMeshRenderer face;
        private bool syncToOthers = true;
        private Vector2 scroll;
        private readonly string[] labels = { "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou", "breath" };
        private readonly string[] shapes = {
            "vrc.v_sil", "vrc.v_pp", "vrc.v_ff", "vrc.v_th", "vrc.v_dd", "vrc.v_kk", "vrc.v_ch", "vrc.v_ss",
            "vrc.v_nn", "vrc.v_rr", "vrc.v_aa", "vrc.v_e", "vrc.v_ih", "vrc.v_oh", "vrc.v_ou", ""
        };

        [MenuItem("Tools/Kotoba LipSync/VRChat FX setup")]
        public static void Open() => GetWindow<KotobaVrcBuilder>("Kotoba VRChat");

        private void OnGUI()
        {
            EditorGUILayout.HelpBox("Creates a new FX controller and expression-parameter asset, preserving the original assets. Avatar Descriptor references are updated with Undo support. Set VRChat OSC on and select the VRChat profile in Kotoba.", MessageType.Info);
            avatar = (GameObject)EditorGUILayout.ObjectField("Avatar root", avatar, typeof(GameObject), true);
            face = (SkinnedMeshRenderer)EditorGUILayout.ObjectField("Face mesh", face, typeof(SkinnedMeshRenderer), true);
            syncToOthers = EditorGUILayout.Toggle("Sync to other players", syncToOthers);
            EditorGUILayout.HelpBox("16 float parameters + Enable = 129 synced bits. Remote expression sync is slower than local lip sync. No custom runtime code runs inside VRChat.", MessageType.None);
            scroll = EditorGUILayout.BeginScrollView(scroll);
            for (int i = 0; i < shapes.Length; i++) shapes[i] = EditorGUILayout.TextField(labels[i], shapes[i]);
            EditorGUILayout.EndScrollView();
            if (GUILayout.Button("Create and assign FX assets"))
            {
                try { Build(); }
                catch (Exception e) { Debug.LogException(e); EditorUtility.DisplayDialog("Kotoba", e.Message, "OK"); }
            }
        }

        private static Type FindType(string name) => AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType(name)).FirstOrDefault(t => t != null);

        private void Build()
        {
            Type descriptorType = FindType("VRC.SDK3.Avatars.Components.VRCAvatarDescriptor");
            Type parameterType = FindType("VRC.SDK3.Avatars.ScriptableObjects.VRCExpressionParameters");
            if (descriptorType == null || parameterType == null) throw new InvalidOperationException("Install VRChat SDK3 Avatars in this Unity project first.");
            if (!avatar || !face || !face.sharedMesh || !face.transform.IsChildOf(avatar.transform)) throw new InvalidOperationException("Select an avatar and its face mesh.");
            Component descriptor = avatar.GetComponent(descriptorType);
            if (!descriptor) throw new InvalidOperationException("Avatar root has no VRCAvatarDescriptor.");
            string meshPath = AnimationUtility.CalculateTransformPath(face.transform, avatar.transform);
            var found = new List<int>();
            for (int i = 0; i < shapes.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(shapes[i])) continue;
                if (face.sharedMesh.GetBlendShapeIndex(shapes[i]) < 0) throw new InvalidOperationException("Missing blendshape: " + shapes[i]);
                found.Add(i);
            }
            if (!found.Any(i => i >= 10 && i <= 14)) throw new InvalidOperationException("Assign the vowel blendshapes before building.");
            if (found.Select(i => shapes[i]).Distinct().Count() != found.Count) throw new InvalidOperationException("Each viseme must use a distinct blendshape.");
            var so = new SerializedObject(descriptor);
            var parametersProperty = so.FindProperty("expressionParameters");
            var oldParameters = parametersProperty.objectReferenceValue;
            ScriptableObject parameters = oldParameters ? Instantiate((ScriptableObject)oldParameters) : CreateInstance(parameterType);
            parameters.name = "KotobaParameters";
            var ps = new SerializedObject(parameters);
            var array = ps.FindProperty("parameters");
            var existing = new List<(string name, int type, bool synced)>();
            for (int i = 0; i < array.arraySize; i++)
            {
                var item = array.GetArrayElementAtIndex(i);
                string name = item.FindPropertyRelative("name").stringValue;
                if (!name.StartsWith("Kotoba_", StringComparison.Ordinal)) existing.Add((name, item.FindPropertyRelative("valueType").enumValueIndex, item.FindPropertyRelative("networkSynced")?.boolValue ?? true));
            }
            int oldCost = existing.Where(x => x.synced && !string.IsNullOrEmpty(x.name)).Sum(x => x.type == 2 ? 1 : 8);
            if (syncToOthers && oldCost + 129 > 256) { DestroyImmediate(parameters); throw new InvalidOperationException($"Synced parameter budget would be {oldCost + 129}/256 bits. Disable sync or free existing parameters."); }
            // Retain all user parameters and remove only a previous Kotoba setup.
            for (int i = array.arraySize - 1; i >= 0; i--)
                if (array.GetArrayElementAtIndex(i).FindPropertyRelative("name").stringValue.StartsWith("Kotoba_", StringComparison.Ordinal)) array.DeleteArrayElementAtIndex(i);
            foreach (string label in labels.Concat(new[] { "Enable" }))
            {
                int n = array.arraySize++;
                var item = array.GetArrayElementAtIndex(n);
                item.FindPropertyRelative("name").stringValue = "Kotoba_" + label;
                item.FindPropertyRelative("valueType").enumValueIndex = label == "Enable" ? 2 : 1; // Int=0, Float=1, Bool=2
                item.FindPropertyRelative("defaultValue").floatValue = label == "sil" ? 1 : 0;
                item.FindPropertyRelative("saved").boolValue = false;
                var synced = item.FindPropertyRelative("networkSynced");
                if (synced != null) synced.boolValue = syncToOthers;
            }
            ps.ApplyModifiedPropertiesWithoutUndo();
            string folder = "Assets/" + "KotobaGenerated_" + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            AssetDatabase.CreateFolder("Assets", folder.Substring("Assets/".Length));
            AssetDatabase.CreateAsset(parameters, folder + "/Parameters.asset");
            var layers = so.FindProperty("baseAnimationLayers");
            SerializedProperty fx = null;
            for (int i = 0; i < layers.arraySize; i++)
            {
                var item = layers.GetArrayElementAtIndex(i);
                if (item.FindPropertyRelative("type").enumNames[item.FindPropertyRelative("type").enumValueIndex] == "FX") fx = item;
            }
            if (fx == null) throw new InvalidOperationException("FX layer was not found on Avatar Descriptor.");
            string controllerPath = folder + "/FX.controller";
            var oldController = fx.FindPropertyRelative("animatorController").objectReferenceValue as RuntimeAnimatorController;
            AnimatorController controller;
            if (oldController)
            {
                if (!(oldController is AnimatorController)) throw new InvalidOperationException("AnimatorOverrideController is not supported. Assign a normal FX controller first.");
                if (!AssetDatabase.CopyAsset(AssetDatabase.GetAssetPath(oldController), controllerPath)) throw new InvalidOperationException("Cannot copy the existing FX controller.");
                controller = AssetDatabase.LoadAssetAtPath<AnimatorController>(controllerPath);
            }
            else controller = AnimatorController.CreateAnimatorControllerAtPath(controllerPath);
            for (int i = controller.layers.Length-1; i >= 0; i--) if (controller.layers[i].name == "Kotoba LipSync") controller.RemoveLayer(i);
            foreach (string label in labels.Concat(new[] { "Enable" }))
            {
                string name = "Kotoba_" + label;
                var expected = label == "Enable" ? AnimatorControllerParameterType.Bool : AnimatorControllerParameterType.Float;
                var existingParameter = controller.parameters.FirstOrDefault(p => p.name == name);
                if (existingParameter != null && existingParameter.type != expected) throw new InvalidOperationException("Animator parameter type conflict: " + name);
                if (existingParameter == null) controller.AddParameter(name, expected);
            }
            controller.AddLayer("Kotoba LipSync");
            var last = controller.layers;
            var layer = last[last.Length-1];
            layer.defaultWeight = 1;
            var idle = layer.stateMachine.AddState("Disabled");
            idle.writeDefaultValues = false;
            var neutral = new AnimationClip { name = "Kotoba neutral" };
            foreach (int j in found)
                neutral.SetCurve(meshPath, typeof(SkinnedMeshRenderer), "blendShape." + shapes[j], AnimationCurve.Constant(0, 1, 0));
            AssetDatabase.CreateAsset(neutral, folder + "/Neutral.anim");
            idle.motion = neutral;
            var active = layer.stateMachine.AddState("Kotoba active");
            active.writeDefaultValues = false;
            var tree = new BlendTree { name = "Kotoba direct visemes", blendType = BlendTreeType.Direct };
            AssetDatabase.AddObjectToAsset(tree, controller);
            foreach (int i in found)
            {
                var clip = new AnimationClip { name = "Kotoba_" + labels[i] };
                // All clips explicitly animate all driven shapes. Direct weights sum poses.
                foreach (int j in found)
                    clip.SetCurve(meshPath, typeof(SkinnedMeshRenderer), "blendShape." + shapes[j], AnimationCurve.Constant(0, 1, i == j ? 100 : 0));
                AssetDatabase.CreateAsset(clip, folder + "/" + labels[i] + ".anim");
                tree.AddChild(clip);
                var children = tree.children;
                children[children.Length-1].directBlendParameter = "Kotoba_" + labels[i];
                tree.children = children;
            }
            var treeSo = new SerializedObject(tree);
            var normalized = treeSo.FindProperty("m_NormalizedBlendValues");
            if (normalized != null) normalized.boolValue = false;
            treeSo.ApplyModifiedPropertiesWithoutUndo();
            active.motion = tree;
            layer.stateMachine.defaultState = idle;
            var enter = idle.AddTransition(active); enter.hasExitTime = false; enter.duration = .02f; enter.AddCondition(AnimatorConditionMode.If, 0, "Kotoba_Enable");
            var exit = active.AddTransition(idle); exit.hasExitTime = false; exit.duration = .04f; exit.AddCondition(AnimatorConditionMode.IfNot, 0, "Kotoba_Enable");
            last[last.Length-1] = layer;
            controller.layers = last;
            Undo.RecordObject(descriptor, "Assign Kotoba LipSync");
            fx.FindPropertyRelative("isDefault").boolValue = false;
            fx.FindPropertyRelative("animatorController").objectReferenceValue = controller;
            parametersProperty.objectReferenceValue = parameters;
            so.FindProperty("customExpressions").boolValue = true;
            so.FindProperty("customizeAnimationLayers").boolValue = true;
            var lip = so.FindProperty("lipSync");
            if (lip != null)
            {
                int parameterOnly = Array.IndexOf(lip.enumNames, "VisemeParameterOnly");
                if (parameterOnly >= 0) lip.enumValueIndex = parameterOnly;
            }
            so.ApplyModifiedProperties();
            EditorUtility.SetDirty(descriptor);
            AssetDatabase.SaveAssets();
            Selection.activeObject = controller;
            EditorUtility.DisplayDialog("Kotoba", "Created " + folder + ". Build & Test the avatar, enable OSC, then start the VRChat output profile.", "OK");
        }
    }
}
