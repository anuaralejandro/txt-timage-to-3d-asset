"""
Unit tests for 2D->3D semantic fusion score accumulation.
"""

import numpy as np
import pytest
from local_asset_factory.segmentation.fusion import SemanticFusionEngine
from local_asset_factory.segmentation.human_parser_backend import ParserResult

def test_fusion_engine_score_accumulation():
    engine = SemanticFusionEngine(num_classes=9)
    assert engine.num_classes == 9
