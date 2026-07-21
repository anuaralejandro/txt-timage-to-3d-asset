import json
from pathlib import Path
import pytest

WORKFLOWS_DIR = Path("workflows")

def test_workflows_exist_and_not_empty():
    workflows = list(WORKFLOWS_DIR.glob("*.json"))
    assert len(workflows) > 0, "No workflows found"
    for wf in workflows:
        assert wf.stat().st_size > 0, f"Workflow {wf.name} is empty (0 bytes)"

def test_workflows_valid_json():
    workflows = list(WORKFLOWS_DIR.glob("*.json"))
    for wf in workflows:
        try:
            with open(wf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {wf.name}: {e}")

def test_hunyuan3d_workflows_have_required_nodes():
    required_nodes = [
        "EmptyLatentHunyuan3Dv2",
        "Hunyuan3Dv2ConditioningMultiView",
        "VAEDecodeHunyuan3D",
        "voxel_to_mesh"
    ]
    
    hunyuan_workflows = [
        "hunyuan3d_2mv_standard.json",
        "hunyuan3d_2mv_turbo.json",
        "hunyuan3d_shape_only_8gb.json"
    ]
    
    for wf_name in hunyuan_workflows:
        wf_path = WORKFLOWS_DIR / wf_name
        if not wf_path.exists():
            continue
            
        with open(wf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Get all class_types in the workflow
        class_types = set()
        for node_id, node_data in data.items():
            if "class_type" in node_data:
                class_types.add(node_data["class_type"])
                
        for req in required_nodes:
            assert req in class_types, f"Node {req} is missing in {wf_name}"

def test_turbo_standard_separation():
    standard_path = WORKFLOWS_DIR / "hunyuan3d_2mv_standard.json"
    turbo_path = WORKFLOWS_DIR / "hunyuan3d_2mv_turbo.json"
    
    if not standard_path.exists() or not turbo_path.exists():
        pytest.skip("Workflows not found")
        
    with open(standard_path, "r", encoding="utf-8") as f:
        std_data = json.load(f)
        
    with open(turbo_path, "r", encoding="utf-8") as f:
        turbo_data = json.load(f)
        
    def get_sampler_node(data):
        for k, v in data.items():
            if v.get("class_type") == "Hunyuan3Dv2Sampler":
                return v
        return None
        
    std_sampler = get_sampler_node(std_data)
    turbo_sampler = get_sampler_node(turbo_data)
    
    assert std_sampler is not None
    assert turbo_sampler is not None
    
    assert std_sampler["inputs"]["subfolder"] == "hunyuan3d-dit-v2-mv"
    assert turbo_sampler["inputs"]["subfolder"] == "hunyuan3d-dit-v2-mv-turbo"
    assert turbo_sampler["inputs"]["guidance_scale"] == 1.0
