import os
import sys
import unittest
import yaml
from pathlib import Path

# Agregar src al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.local_asset_factory.multiview.canonical_views import build_canonical_view_set
from src.local_asset_factory.scoring.mesh_scoring import check_hard_gates
from src.local_asset_factory.parts3d.segmentation import SemanticSegmenter

class TestE2EPipeline(unittest.TestCase):
    
    def test_configs_exist(self):
        root = Path(__file__).parent.parent
        normal = root / "configs" / "models" / "hunyuan3d_2mv_normal.yaml"
        turbo = root / "configs" / "models" / "hunyuan3d_2mv_turbo.yaml"
        
        self.assertTrue(normal.exists())
        self.assertTrue(turbo.exists())
        
        with open(normal) as f:
            n_cfg = yaml.safe_load(f)
            self.assertEqual(n_cfg["variant"], "normal")
            
        with open(turbo) as f:
            t_cfg = yaml.safe_load(f)
            self.assertEqual(t_cfg["variant"], "turbo")
            
    def test_quality_gates(self):
        # A good candidate
        good = {
            "silhouette_iou": 0.85,
            "connected_components": 1,
            "has_head": 1,
            "arms_count": 2,
            "legs_count": 2,
            "severe_cavities": 0
        }
        self.assertEqual(len(check_hard_gates(good)), 0)
        
        # A bad candidate (no head)
        bad = good.copy()
        bad["has_head"] = 0
        self.assertIn("head_missing", check_hard_gates(bad))
        
    def test_segmentation_mock(self):
        seg = SemanticSegmenter()
        parts = seg.segment_mesh("raw_mesh.glb")
        self.assertIn("base_body", parts)
        self.assertIn("clothing_top", parts)

if __name__ == '__main__':
    unittest.main()
