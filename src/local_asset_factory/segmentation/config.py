"""
local_asset_factory · segmentation · config
Configuration dataclasses for low-VRAM 8GB execution.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class SegmentationConfig:
    """Settings optimized for 8GB VRAM execution (NVIDIA RTX 4070 Laptop)."""
    low_vram_mode: bool = True
    render_resolution: int = 768
    render_views: int = 8  # 6 or 8 orthographic views
    batch_size: int = 1
    dtype: str = "float16"
    
    # Model preferences
    sam2_model: str = "sam2.1_hiera_small"
    parser_model: str = "schp_lip"
    pose_model: str = "dwpose"
    enable_p3sam: bool = True
    p3sam_point_count: int = 50000
    
    # Mesh decimation proxy thresholds
    proxy_triangle_threshold: int = 250000
    proxy_target_triangles: int = 150000
    
    # Postprocessing parameters
    min_island_faces: int = 25
    smoothing_iterations: int = 2
    merge_head_and_hair: bool = False  # Set to True if head & hair geometry are tightly merged on the mesh

    
    # Texture baking
    texture_resolution: int = 1024
    uv_dilation_px: int = 12
    enable_texture_projection: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "low_vram_mode": self.low_vram_mode,
            "render_resolution": self.render_resolution,
            "render_views": self.render_views,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "sam2_model": self.sam2_model,
            "parser_model": self.parser_model,
            "pose_model": self.pose_model,
            "enable_p3sam": self.enable_p3sam,
            "p3sam_point_count": self.p3sam_point_count,
            "proxy_triangle_threshold": self.proxy_triangle_threshold,
            "proxy_target_triangles": self.proxy_target_triangles,
            "min_island_faces": self.min_island_faces,
            "smoothing_iterations": self.smoothing_iterations,
            "texture_resolution": self.texture_resolution,
            "uv_dilation_px": self.uv_dilation_px,
            "enable_texture_projection": self.enable_texture_projection,
        }
