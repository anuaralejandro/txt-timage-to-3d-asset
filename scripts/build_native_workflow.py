import json
import os

def create_native_hunyuan_multiview_workflow():
    workflow = {
        "last_node_id": 20,
        "last_link_id": 35,
        "nodes": [
            {
                "id": 1,
                "type": "AssetFactory_LoadMultiviewDirectory",
                "pos": [50, 100],
                "size": [350, 250],
                "widgets_values": [False, "./input/character"],
                "_meta": {"title": "1 · Load Multiview Directory"}
            },
            {
                "id": 2,
                "type": "AssetFactory_RemoveMultiviewFakeBackground",
                "pos": [450, 100],
                "size": [350, 160],
                "inputs": [
                    {"name": "FRONT_IMAGE", "type": "IMAGE", "link": 1},
                    {"name": "LEFT_IMAGE", "type": "IMAGE", "link": 2},
                    {"name": "RIGHT_IMAGE", "type": "IMAGE", "link": 3},
                    {"name": "BACK_IMAGE", "type": "IMAGE", "link": 4}
                ],
                "widgets_values": [False],
                "_meta": {"title": "2 · Remove Multiview Background"}
            },
            {
                "id": 3,
                "type": "CLIPVisionLoader",
                "pos": [50, 400],
                "size": [315, 80],
                "widgets_values": ["dinov2_giant.safetensors"],
                "_meta": {"title": "3 · Load DinoV2 (CLIP Vision Loader)"}
            },
            {
                "id": 4,
                "type": "CLIPVisionEncode",
                "pos": [450, 320],
                "size": [210, 80],
                "inputs": [
                    {"name": "clip_vision", "type": "CLIP_VISION", "link": 5},
                    {"name": "image", "type": "IMAGE", "link": 6}
                ],
                "_meta": {"title": "4 · Encode Front View"}
            },
            {
                "id": 5,
                "type": "CLIPVisionEncode",
                "pos": [450, 430],
                "size": [210, 80],
                "inputs": [
                    {"name": "clip_vision", "type": "CLIP_VISION", "link": 5},
                    {"name": "image", "type": "IMAGE", "link": 7}
                ],
                "_meta": {"title": "5 · Encode Left View"}
            },
            {
                "id": 6,
                "type": "CLIPVisionEncode",
                "pos": [450, 540],
                "size": [210, 80],
                "inputs": [
                    {"name": "clip_vision", "type": "CLIP_VISION", "link": 5},
                    {"name": "image", "type": "IMAGE", "link": 9}
                ],
                "_meta": {"title": "6 · Encode Back View"}
            },
            {
                "id": 7,
                "type": "CLIPVisionEncode",
                "pos": [450, 650],
                "size": [210, 80],
                "inputs": [
                    {"name": "clip_vision", "type": "CLIP_VISION", "link": 5},
                    {"name": "image", "type": "IMAGE", "link": 8}
                ],
                "_meta": {"title": "7 · Encode Right View"}
            },
            {
                "id": 8,
                "type": "Hunyuan3Dv2ConditioningMultiView",
                "pos": [700, 320],
                "size": [300, 150],
                "inputs": [
                    {"name": "front", "type": "CLIP_VISION_OUTPUT", "link": 10},
                    {"name": "left", "type": "CLIP_VISION_OUTPUT", "link": 11},
                    {"name": "back", "type": "CLIP_VISION_OUTPUT", "link": 12},
                    {"name": "right", "type": "CLIP_VISION_OUTPUT", "link": 13}
                ],
                "_meta": {"title": "8 · MultiView Conditioning"}
            },
            {
                "id": 9,
                "type": "CheckpointLoaderSimple",
                "pos": [700, 100],
                "size": [315, 100],
                "widgets_values": ["hunyuan3d-dit-v2-mv-turbo.safetensors"],
                "_meta": {"title": "9 · Load Hunyuan3D-2mv Checkpoint (Normal / Turbo)"}
            },
            {
                "id": 10,
                "type": "EmptyLatentHunyuan3Dv2",
                "pos": [700, 500],
                "size": [300, 90],
                "widgets_values": [3072, 1],
                "_meta": {"title": "10 · Empty Hunyuan Latent"}
            },
            {
                "id": 11,
                "type": "KSampler",
                "pos": [1050, 100],
                "size": [315, 260],
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": 14},
                    {"name": "positive", "type": "CONDITIONING", "link": 15},
                    {"name": "negative", "type": "CONDITIONING", "link": 16},
                    {"name": "latent_image", "type": "LATENT", "link": 17}
                ],
                "widgets_values": [42, "randomize", 50, 5.5, "euler", "normal", 1.0],
                "_meta": {"title": "11 · KSampler (Steps / Sampler / CFG Control)"}
            },
            {
                "id": 12,
                "type": "VAELoader",
                "pos": [1050, 400],
                "size": [315, 80],
                "widgets_values": ["hunyuan3d-vae-v2-0-turbo.safetensors"],
                "_meta": {"title": "12 · Load Hunyuan VAE"}
            },
            {
                "id": 13,
                "type": "VAEDecodeHunyuan3D",
                "pos": [1400, 100],
                "size": [280, 120],
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": 18},
                    {"name": "vae", "type": "VAE", "link": 19}
                ],
                "widgets_values": [8000, 256],
                "_meta": {"title": "13 · VAE Decode to Voxels"}
            },
            {
                "id": 14,
                "type": "VoxelToMesh",
                "pos": [1400, 260],
                "size": [280, 120],
                "inputs": [
                    {"name": "voxel", "type": "VOXEL", "link": 20}
                ],
                "widgets_values": ["surface net", 0.5],
                "_meta": {"title": "14 · Voxel to Mesh (Algorithm & Threshold)"}
            },
            {
                "id": 15,
                "type": "AssetFactory_SaveMeshToGLB",
                "pos": [1720, 100],
                "size": [300, 100],
                "inputs": [
                    {"name": "mesh", "type": "MESH", "link": 21}
                ],
                "widgets_values": ["hunyuan3d_character"],
                "_meta": {"title": "15 · Save Mesh to GLB"}
            },
            {
                "id": 16,
                "type": "AssetFactory_Preview3D",
                "pos": [2100, 100],
                "size": [350, 400],
                "inputs": [
                    {"name": "model_path", "type": "STRING", "link": 23}
                ],
                "_meta": {"title": "16 · 3D Mesh Canvas Preview"}
            },
            {
                "id": 17,
                "type": "AssetFactory_BlenderProcessor",
                "pos": [1720, 230],
                "size": [315, 260],
                "inputs": [
                    {"name": "MODEL_PATH", "type": "STRING", "link": 22}
                ],
                "widgets_values": ["", True, True, True, True, False, True],
                "_meta": {"title": "17 · Blender Mesh Smoothing & Optimize"}
            }
        ],
        "links": [
            [1, 1, 0, 2, 0, "IMAGE"],
            [2, 1, 1, 2, 1, "IMAGE"],
            [3, 1, 2, 2, 2, "IMAGE"],
            [4, 1, 3, 2, 3, "IMAGE"],
            [5, 3, 0, 4, 0, "CLIP_VISION"],
            [5, 3, 0, 5, 0, "CLIP_VISION"],
            [5, 3, 0, 6, 0, "CLIP_VISION"],
            [5, 3, 0, 7, 0, "CLIP_VISION"],
            [6, 2, 0, 4, 1, "IMAGE"],
            [7, 2, 1, 5, 1, "IMAGE"],
            [8, 2, 2, 7, 1, "IMAGE"],
            [9, 2, 3, 6, 1, "IMAGE"],
            [10, 4, 0, 8, 0, "CLIP_VISION_OUTPUT"],
            [11, 5, 0, 8, 1, "CLIP_VISION_OUTPUT"],
            [12, 6, 0, 8, 2, "CLIP_VISION_OUTPUT"],
            [13, 7, 0, 8, 3, "CLIP_VISION_OUTPUT"],
            [14, 9, 0, 11, 0, "MODEL"],
            [15, 8, 0, 11, 1, "CONDITIONING"],
            [16, 8, 1, 11, 2, "CONDITIONING"],
            [17, 10, 0, 11, 3, "LATENT"],
            [18, 11, 0, 13, 0, "LATENT"],
            [19, 12, 0, 13, 1, "VAE"],
            [20, 13, 0, 14, 0, "VOXEL"],
            [21, 14, 0, 15, 0, "MESH"],
            [22, 15, 0, 17, 0, "STRING"],
            [23, 17, 0, 16, 0, "STRING"]
        ],
        "version": 0.4
    }

    # Auto-generate outputs array for GUI compatibility
    nodes_by_id = {n["id"]: n for n in workflow["nodes"]}
    for link in workflow["links"]:
        link_id, from_node, from_slot, to_node, to_slot, type_str = link
        node = nodes_by_id.get(from_node)
        if node:
            if "outputs" not in node:
                node["outputs"] = []
            
            # Find if output slot already exists
            out_slot = next((o for o in node["outputs"] if o.get("slot_index") == from_slot), None)
            if not out_slot:
                out_slot = {
                    "name": type_str,
                    "type": type_str,
                    "links": [],
                    "slot_index": from_slot
                }
                node["outputs"].append(out_slot)
            
            out_slot["links"].append(link_id)

    # Sort outputs by slot_index
    for n in workflow["nodes"]:
        if "outputs" in n:
            n["outputs"].sort(key=lambda x: x["slot_index"])

    target_paths = [
        "workflows/04_hunyuan_multiview_unified.json",
        "comfyui/ComfyUI-LocalAssetFactory/workflows/04_hunyuan_multiview_unified.json",
        "ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-LocalAssetFactory/workflows/04_hunyuan_multiview_unified.json"
    ]

    for p in target_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2)
        print(f"Updated: {p}")

if __name__ == "__main__":
    create_native_hunyuan_multiview_workflow()
