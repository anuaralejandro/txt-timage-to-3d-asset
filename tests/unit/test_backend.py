import pytest
from services.hunyuan3d_2mv.backend import Hunyuan2MVBackend

def test_backend_capabilities():
    backend = Hunyuan2MVBackend()
    caps = backend.capabilities()
    assert "front" in caps.supported_views
    assert "left" in caps.supported_views
    assert "back" in caps.supported_views
    assert "right" not in caps.supported_views # The model itself does not take 'right', our wrapper filters it.

def test_backend_generate_retains_right_view(monkeypatch):
    backend = Hunyuan2MVBackend()
    
    # Mock load and model to avoid loading real weights
    def mock_load(*args, **kwargs):
        pass
    backend._load_model = mock_load
    
    class MockModel:
        def __call__(self, *args, **kwargs):
            class DummyMesh:
                def __init__(self):
                    self.vertices = [[0,0,0], [1,0,0], [0,1,0]]
                    self.faces = [[0,1,2]]
                def export(self, path):
                    with open(path, "w") as f:
                        f.write("mock")
            return DummyMesh()
        def enable_model_cpu_offload(self):
            pass
        
    backend._model = MockModel()
    
    # Test that 'right' in views doesn't crash it and is recorded in manifest
    import tempfile
    from pathlib import Path
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        from PIL import Image
        for v in ["front", "back", "left", "right"]:
            img = Image.new("RGBA", (10, 10))
            img.save(Path(tmpdir) / f"{v}.png")
            
        views = {v: str(Path(tmpdir) / f"{v}.png") for v in ["front", "back", "left", "right"]}
        
        result = backend.generate(
            views=views,
            model_id="dummy",
            model_variant="normal",
            output_dir=tmpdir,
            job_id="test"
        )
        
        assert result["status"] == "success"
        
        # Check manifest
        manifest_path = Path(result["input_manifest_path"])
        assert manifest_path.exists()
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        assert "right" in manifest["views_provided"]
        assert "right" not in manifest["views_used_by_model"]
        assert "front" in manifest["views_used_by_model"]
