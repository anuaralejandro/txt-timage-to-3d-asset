import urllib.request
import json

prompt = {
    "1": {
        "class_type": "AssetFactory_LocalHunyuan",
        "inputs": {
            "multiview_folder": "./input/jace",
            "workflow_path": "",
            "generate_3d": True,
            "resolution": 1024,
            "steps": 10,
            "cfg": 5.0,
            "mesh_algorithm": "surface net",
            "mesh_threshold": 0.6,
            "seed": 42,
            "timeout_seconds": 600
        }
    },
    "2": {
        "class_type": "AssetFactory_SaveManifest",
        "inputs": {
            "MODEL_PATH": ["1", 0],
            "ASSET_DIR": ["1", 1]
        }
    }
}

p = {"prompt": prompt, "client_id": "test_script"}
data = json.dumps(p).encode('utf-8')
req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data)
try:
    with urllib.request.urlopen(req) as response:
        print("Success:", response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print(f"Error {e.code}: {e.read().decode('utf-8')}")
except Exception as e:
    print("Error:", e)
