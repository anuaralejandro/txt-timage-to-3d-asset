"""
Unit and Integration Tests for the Sapiens2 + P3-SAM + Skeleton 3D Anatomical Segmentation Pipeline.
"""

import os
import sys
import unittest
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path

# Ensure src is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from local_asset_factory.segmentation import (
    SemanticLabel,
    LABEL_NAMES,
    FaceViewRenderer,
    SapiensSegmentor,
    GeometryJointPredictor,
    compute_bone_priors,
    fuse_anatomical_signals,
    refine_labels_mrf,
    transfer_labels_proxy_to_full,
)

class TestSapiensPipeline(unittest.TestCase):

    def setUp(self):
        # Create a simple synthetic cylinder humanoid mesh for testing
        self.mesh = trimesh.creation.capsule(height=2.0, radius=0.4)
        self.output_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(self.output_dir, exist_ok=True)

    def test_face_view_renderer(self):
        renderer = FaceViewRenderer(use_blender=False)
        views = renderer.render_views(self.mesh, self.output_dir, view_count=4, resolution=256)
        self.assertEqual(len(views), 4)
        self.assertTrue(os.path.isfile(views[0]["face_id_path"]))

    def test_sapiens_segmentor_synthetic(self):
        segmentor = SapiensSegmentor()
        dummy_rgb = os.path.join(self.output_dir, "dummy_rgb.png")
        Image.new("RGB", (256, 256), (200, 200, 200)).save(dummy_rgb)
        res = segmentor._synthetic_sapiens_predict(Image.open(dummy_rgb).convert("RGB"), is_front=True)
        self.assertEqual(res.logits.shape[2], 18)
        self.assertEqual(res.labels.shape, (res.logits.shape[0], res.logits.shape[1]))

    def test_skeleton_predictor_and_priors(self):
        predictor = GeometryJointPredictor()
        joints = predictor.predict(self.mesh)
        self.assertIn("head", joints)
        self.assertIn("pelvis", joints)
        priors = compute_bone_priors(self.mesh, joints)
        self.assertEqual(priors.shape, (len(self.mesh.faces), 18))

    def test_fusion_and_mrf_refinement(self):
        num_faces = len(self.mesh.faces)
        sapiens_probs = np.full((num_faces, 18), 1.0 / 18)
        p3sam_regions = np.zeros(num_faces, dtype=np.int32)
        bone_priors = np.full((num_faces, 18), 1.0 / 18)

        unary = fuse_anatomical_signals(sapiens_probs, p3sam_regions, bone_priors)
        self.assertEqual(unary.shape, (num_faces, 18))

        refined = refine_labels_mrf(self.mesh, unary, p3sam_regions, max_iterations=2)
        self.assertEqual(len(refined), num_faces)

    def test_proxy_to_full_transfer(self):
        proxy = self.mesh.copy()
        proxy_labels = np.zeros(len(proxy.faces), dtype=np.int32)
        full_labels = transfer_labels_proxy_to_full(proxy, proxy_labels, self.mesh)
        self.assertEqual(len(full_labels), len(self.mesh.faces))

if __name__ == "__main__":
    unittest.main()
