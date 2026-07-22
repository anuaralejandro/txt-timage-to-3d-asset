import pytest
from unittest.mock import MagicMock
from local_asset_factory.preflight.preflight_runner import PreflightRunner, PreflightConfig
from local_asset_factory.geometry.hunyuan2mv_client import Hunyuan2MVClient
from local_asset_factory.scoring.ranker import rank_candidates
from services.hunyuan3d_paint.backend import HunyuanPaintBackend

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.path.return_value = "/tmp/mock/path"
    return store

def test_pipeline_halts_on_preflight_failure():
    runner = PreflightRunner(PreflightConfig(require_front=True))
    views = {"back": "dummy.png"} # missing front
    
    results = runner.validate_view_set(views)
    assert not runner.all_passed(results)
    
    # In a real orchestrator, this failure means we don't proceed to shape generation.

def test_pipeline_no_success_on_empty_mesh(mock_store):
    backend = MagicMock()
    backend.capabilities().supported_views = ["front"]
    backend.capabilities().checkpoint = "dummy"
    # Mocking a backend that returns an error
    backend.generate.return_value = {
        "status": "error",
        "mesh_path": "",
        "warnings": ["empty mesh"]
    }
    
    scheduler = MagicMock()
    scheduler.slot.return_value.__enter__ = MagicMock()
    scheduler.slot.return_value.__exit__ = MagicMock()
    
    client = Hunyuan2MVClient(backend, scheduler)
    
    request = MagicMock()
    request.views = {"front": "dummy.png"}
    request.job_id = "test_job"
    
    cand = client.generate_candidate(request, mock_store)
    
    assert not cand.passed_gates
    assert "error" in cand.warnings[0]

def test_pipeline_skips_paint_when_disabled():
    paint = HunyuanPaintBackend(enabled=False)
    res = paint.paint_mesh("dummy.glb", {}, "/tmp/out", "job1")
    assert res["status"] == "unsupported_on_current_hardware"
    
def test_pipeline_fails_when_all_candidates_fail():
    c1 = MagicMock(passed_gates=False, id="1")
    c1.score = 0.0
    c2 = MagicMock(passed_gates=False, id="2")
    c2.score = 0.0
    cands = [c1, c2]
    winner, _ = rank_candidates(cands)
    assert winner is None
