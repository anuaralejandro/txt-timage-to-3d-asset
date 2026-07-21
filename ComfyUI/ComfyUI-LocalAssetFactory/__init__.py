"""
ComfyUI-LocalAssetFactory
=========================

Hunyuan multiview character pipeline for game development.
Transforms multiview images into rigged, low-poly 3D characters using:
- SAM 3.1 (semantic 2D segmentation)
- Hunyuan3D-2mv (multiview geometry backbone)
- Hunyuan3D-Omni (pose-controlled generation)
- Hunyuan3D-Part (3D part segmentation)
- Hunyuan3D-Paint (post-UV texturing)
- Blender (retopology, UV, rigging, export)

NOTE: TRELLIS has been removed. This is a Hunyuan-only pipeline.
All backends are local-first. No internet required if models are pre-downloaded.
"""

import sys
import os
from pathlib import Path

# Add project src to sys.path so ComfyUI can find local_asset_factory
_ROOT = Path(__file__).parent.parent.parent.parent.parent
_SRC_DIR = _ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from .nodes import (
    AssetBriefNode,
    OllamaPromptArchitectNode,
    LocalConceptImageNode,
    LocalTextureGeneratorNode,
    LocalHunyuanNode,
    BlenderProcessorNode,
    SaveManifestNode,
    LocalAssetFactoryPreview3DNode,
)
from .nodes_macro import HunyuanMacroPipelineNode
from .nodes_image_processing import RemoveFakeBackgroundNode, RemoveMultiviewFakeBackgroundNode, ApplyManualMaskNode
from .nodes_io import LoadMultiviewDirectoryNode, SaveMeshToGLBNode

# ── ComfyUI Registration ─────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AssetFactory_AssetBrief": AssetBriefNode,
    "AssetFactory_OllamaPromptArchitect": OllamaPromptArchitectNode,
    "AssetFactory_LocalConceptImage": LocalConceptImageNode,
    "AssetFactory_LocalTextureGenerator": LocalTextureGeneratorNode,
    "AssetFactory_LocalHunyuan": LocalHunyuanNode,
    "AssetFactory_BlenderProcessor": BlenderProcessorNode,
    "AssetFactory_SaveManifest": SaveManifestNode,
    "AssetFactory_Preview3D": LocalAssetFactoryPreview3DNode,
    "AssetFactory_HunyuanMacroPipeline": HunyuanMacroPipelineNode,
    "AssetFactory_RemoveFakeBackground": RemoveFakeBackgroundNode,
    "AssetFactory_RemoveMultiviewFakeBackground": RemoveMultiviewFakeBackgroundNode,
    "AssetFactory_ApplyManualMask": ApplyManualMaskNode,
    "AssetFactory_LoadMultiviewDirectory": LoadMultiviewDirectoryNode,
    "AssetFactory_SaveMeshToGLB": SaveMeshToGLBNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AssetFactory_AssetBrief": "Asset Factory · Asset Brief",
    "AssetFactory_OllamaPromptArchitect": "Asset Factory · Ollama Prompt Architect",
    "AssetFactory_LocalConceptImage": "Asset Factory · Local Concept Image",
    "AssetFactory_LocalTextureGenerator": "Asset Factory · Local Texture Generator",
    "AssetFactory_LocalHunyuan": "Asset Factory · Hunyuan3D-2mv Image-to-3D",
    "AssetFactory_BlenderProcessor": "Asset Factory · Blender Processor",
    "AssetFactory_SaveManifest": "Asset Factory · Save Manifest",
    "AssetFactory_Preview3D": "Asset Factory · 3D GLB Previewer",
    "AssetFactory_HunyuanMacroPipeline": "Asset Factory · Macro Pipeline (M2-M6)",
    "AssetFactory_RemoveFakeBackground": "Asset Factory · Remove Fake Background",
    "AssetFactory_RemoveMultiviewFakeBackground": "Asset Factory · Remove Fake Background (Multi-view)",
    "AssetFactory_ApplyManualMask": "Asset Factory · Apply Manual Mask",
    "AssetFactory_LoadMultiviewDirectory": "Asset Factory · Load Multiview Directory",
    "AssetFactory_SaveMeshToGLB": "Asset Factory · Save Mesh to GLB",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("\033[92m[LocalAssetFactory]\033[0m Loaded 10 nodes (Hunyuan3D-2mv backbone, TRELLIS removed).")
