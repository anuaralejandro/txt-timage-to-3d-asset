import json
import urllib.request
import time
import sys

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

prompt = {
  "1": {
    "inputs": {
      "MODEL_PATH": "C:\\Users\\datam\\Downloads\\jace0_clean_no_droplets.glb",
      "triangle_threshold": 250000,
      "target_triangles": 150000
    },
    "class_type": "AssetFactory_CreateSegmentationProxy",
    "_meta": { "title": "Create Segmentation Proxy" }
  },
  "2": {
    "inputs": {
      "MODEL_PATH": ["1", 0],
      "point_count": 50000,
      "enable": True
    },
    "class_type": "AssetFactory_P3SAMSegment",
    "_meta": { "title": "P3-SAM 3D Segmentation" }
  },
  "3": {
    "inputs": {
      "MODEL_PATH": ["1", 0],
      "view_count": 8,
      "resolution": 768
    },
    "class_type": "AssetFactory_RenderSemanticViews",
    "_meta": { "title": "Render Semantic Views" }
  },
  "4": {
    "inputs": {
      "RENDER_MANIFEST_PATH": ["3", 0],
      "backend_model": "schp_lip"
    },
    "class_type": "AssetFactory_HumanParseViews",
    "_meta": { "title": "Human Parse 2D Views (X-Part)" }
  },
  "5": {
    "inputs": {
      "MODEL_PATH": ["1", 0],
      "RENDER_MANIFEST_PATH": ["3", 0],
      "PARSER_RESULTS_PATH": ["4", 0],
      "P3SAM_RESULTS_PATH": ["2", 0],
      "merge_head_and_hair": False
    },
    "class_type": "AssetFactory_FuseSemanticParts",
    "_meta": { "title": "Fuse Semantic Parts 2D->3D" }
  },
  "6": {
    "inputs": {
      "MODEL_PATH": ["1", 0],
      "FACE_LABELS_PATH": ["5", 0],
      "SEMANTIC_MANIFEST_PATH": ["5", 1]
    },
    "class_type": "AssetFactory_WriteSemanticGLB",
    "_meta": { "title": "Write Semantic GLB" }
  },
  "7": {
    "inputs": {
        "PROCESSED_MODEL_PATH": ["6", 0]
    },
    "class_type": "AssetFactory_SaveManifest",
    "_meta": { "title": "Trigger Output Node" }
  }
}

print("Submitting P3-SAM & X-Part segmentation workflow to local ComfyUI...")
max_retries = 3
response = None
for i in range(max_retries):
    try:
        response = queue_prompt(prompt)
        print("Response:", response)
        break
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode()}")
        sys.exit(1)
    except urllib.error.URLError as e:
        if i == max_retries - 1:
            print(f"Failed to connect to ComfyUI after {max_retries} attempts: {e}")
            sys.exit(1)
        time.sleep(2)
        print(f"Waiting for ComfyUI server... ({i+1}/{max_retries})")
    except Exception as e:
        print(f"Error submitting prompt: {e}")
        sys.exit(1)

if response and 'prompt_id' in response:
    prompt_id = response['prompt_id']
    print(f"Workflow queued! Prompt ID: {prompt_id}")
    print("Please check your ComfyUI terminal window to see the execution progress!")
