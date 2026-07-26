"""
local_asset_factory · segmentation
3D Anatomical Segmentation and Multiview Texture Projection package.
"""

from .labels import SemanticLabel, LABEL_NAMES, LABEL_COLORS_RGB, resolve_anatomical_side
from .config import SegmentationConfig
from .contracts import SemanticPartsManifest, ViewRenderInfo, PoseDetectionResult
from .memory import ModelLifecycleManager
from .proxy import ProxyManager
from .semantic_renderer import SemanticRenderer
from .human_parser_backend import MockHumanParser, SCHPLIPHumanParser, ParserResult
from .pose_backend import MockPoseBackend, DWPoseBackend
from .sam2_backend import MockSAM2Backend, SAM2HieraSmallBackend
from .p3sam_backend import MockP3SAMBackend, P3SAMSonataBackend
from .fusion import SemanticFusionEngine
from .postprocess import SegmentationPostProcessor
from .neck_inference import NeckInferencer
from .export import SemanticExporter
from .texture_projection import MultiviewTextureProjector
from .face_renderer import FaceViewRenderer, decode_face_id_map
from .sapiens_backend import SapiensSegmentor, SapiensResult
from .skeleton_backend import GeometryJointPredictor, compute_bone_priors
from .graph_optimization import fuse_anatomical_signals, refine_labels_mrf, transfer_labels_proxy_to_full

__all__ = [
    "SemanticLabel",
    "LABEL_NAMES",
    "LABEL_COLORS_RGB",
    "resolve_anatomical_side",
    "SegmentationConfig",
    "SemanticPartsManifest",
    "ViewRenderInfo",
    "PoseDetectionResult",
    "ModelLifecycleManager",
    "ProxyManager",
    "SemanticRenderer",
    "MockHumanParser",
    "SCHPLIPHumanParser",
    "ParserResult",
    "MockPoseBackend",
    "DWPoseBackend",
    "MockSAM2Backend",
    "SAM2HieraSmallBackend",
    "MockP3SAMBackend",
    "P3SAMSonataBackend",
    "SemanticFusionEngine",
    "SegmentationPostProcessor",
    "NeckInferencer",
    "SemanticExporter",
    "MultiviewTextureProjector",
    "FaceViewRenderer",
    "decode_face_id_map",
    "SapiensSegmentor",
    "SapiensResult",
    "GeometryJointPredictor",
    "compute_bone_priors",
    "fuse_anatomical_signals",
    "refine_labels_mrf",
    "transfer_labels_proxy_to_full",
]
