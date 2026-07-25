"""
Integration test for 3D Segmentation & Multiview Texture Projection Pipeline using mock backends.
"""

import os
import tempfile
import numpy as np
import pytest

from local_asset_factory.segmentation import (
    SemanticLabel,
    ProxyManager,
    MockHumanParser,
    MockPoseBackend,
    MockSAM2Backend,
    MockP3SAMBackend,
    SemanticFusionEngine,
    SegmentationPostProcessor,
    NeckInferencer,
    SemanticExporter,
    MultiviewTextureProjector,
)

def test_full_segmentation_pipeline_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_glb = os.path.join(tmpdir, "character.glb")
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.export(fake_glb)
        face_count = len(mesh.faces)

        # 1. Human Parsing
        parser = MockHumanParser()
        parser.load({})
        assert parser.is_loaded

        # 2. Pose
        pose = MockPoseBackend()
        pose.load({})
        assert pose.is_loaded

        # 3. SAM2
        sam2 = MockSAM2Backend()
        sam2.load({})
        assert sam2.is_loaded

        # 4. Initial face labels
        initial_labels = np.random.randint(1, 9, size=face_count)

        # 5. Topology cleaner
        cleaner = SegmentationPostProcessor()
        # 6. Exporter
        exporter = SemanticExporter()
        seg_glb, labels_npz, manifest_json, colored_glb = exporter.export_all(
            source_glb_path=fake_glb,
            output_dir=tmpdir,
            face_labels=initial_labels,
            face_confidences=np.full(face_count, 0.95),
        )

        assert os.path.isfile(seg_glb)
        assert os.path.isfile(labels_npz)
        assert os.path.isfile(manifest_json)
        assert os.path.isfile(colored_glb)
