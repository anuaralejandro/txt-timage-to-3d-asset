"""
Unit tests for proxy decimation and BVH label transfer.
"""

import numpy as np
import pytest
from local_asset_factory.segmentation.proxy import ProxyManager

def test_proxy_manager_threshold_logic():
    manager = ProxyManager(triangle_threshold=250000, target_triangles=150000)
    assert manager.triangle_threshold == 250000
    assert manager.target_triangles == 150000
