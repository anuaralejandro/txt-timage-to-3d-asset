"""
Regression tests to verify workflow 04 intactness and node registration.
"""

import os
import json
import pytest

def test_workflow_04_intactness():
    w4_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "ComfyUI_windows_portable",
        "ComfyUI",
        "custom_nodes",
        "ComfyUI-LocalAssetFactory",
        "workflows",
        "04_hunyuan_multiview_unified.json",
    )
    if not os.path.isfile(w4_path):
        w4_path = os.path.join(os.path.dirname(__file__), "..", "..", "workflows", "04_hunyuan_multiview_unified.json")

    assert os.path.isfile(w4_path)
    with open(w4_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "nodes" in data
    assert "links" in data
    assert len(data["nodes"]) > 10

def test_workflow_05_validity():
    w5_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "ComfyUI_windows_portable",
        "ComfyUI",
        "custom_nodes",
        "ComfyUI-LocalAssetFactory",
        "workflows",
        "05_hunyuan_multiview_segment_and_texture_8gb.json",
    )
    if not os.path.isfile(w5_path):
        w5_path = os.path.join(os.path.dirname(__file__), "..", "..", "workflows", "05_hunyuan_multiview_segment_and_texture_8gb.json")

    assert os.path.isfile(w5_path)
    with open(w5_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "nodes" in data
    assert "links" in data
    assert len(data["nodes"]) == 27

def test_node_registration():
    import sys
    import importlib
    comfyui_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "ComfyUI_windows_portable",
            "ComfyUI",
        )
    )
    if comfyui_dir not in sys.path:
        sys.path.insert(0, comfyui_dir)

    custom_nodes_parent = os.path.join(comfyui_dir, "custom_nodes")
    if custom_nodes_parent not in sys.path:
        sys.path.insert(0, custom_nodes_parent)

    pkg_mod = importlib.import_module("ComfyUI-LocalAssetFactory")
    NODE_CLASS_MAPPINGS = pkg_mod.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = pkg_mod.NODE_DISPLAY_NAME_MAPPINGS

    # Core nodes
    assert "AssetFactory_BlenderProcessor" in NODE_CLASS_MAPPINGS
    assert "AssetFactory_LocalHunyuan" in NODE_CLASS_MAPPINGS

    # 10 New Segmentation nodes
    new_nodes = [
        "AssetFactory_CreateSegmentationProxy",
        "AssetFactory_P3SAMSegment",
        "AssetFactory_RenderSemanticViews",
        "AssetFactory_HumanParseViews",
        "AssetFactory_PoseSemanticHints",
        "AssetFactory_SAM2RefineParts",
        "AssetFactory_FuseSemanticParts",
        "AssetFactory_WriteSemanticGLB",
        "AssetFactory_ProjectMultiviewTexture",
        "AssetFactory_SegmentationPreview",
    ]
    for node_key in new_nodes:
        assert node_key in NODE_CLASS_MAPPINGS, f"Missing node mapping for {node_key}"
        assert node_key in NODE_DISPLAY_NAME_MAPPINGS, f"Missing display name mapping for {node_key}"
