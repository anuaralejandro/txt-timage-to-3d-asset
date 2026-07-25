import json
import urllib.request
import time
import sys

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

# Convert workflow to API format for prompt submission
prompt = {
  "1": {
    "inputs": {
      "glb_path": "C:\\Users\\datam\\Downloads\\jace0_clean_no_droplets.glb",
      "vram_optimization": "8GB_Turbo"
    },
    "class_type": "AssetFactory_Hunyuan3DPartSegment",
    "_meta": {
      "title": "Hunyuan3D-Part GPU 3D Segmentation"
    }
  },
  "2": {
    "inputs": {
      "model_path": [
        "1",
        1
      ]
    },
    "class_type": "AssetFactory_Preview3D",
    "_meta": {
      "title": "3D Previewer"
    }
  }
}

print("Submitting local GPU workflow to ComfyUI...")
max_retries = 30
response = None
for i in range(max_retries):
    try:
        response = queue_prompt(prompt)
        print("Response:", response)
        break
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
    while True:
        try:
            history = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/history").read())
            if prompt_id in history:
                print("=== Local GPU Execution Completed Successfully ===")
                print(json.dumps(history[prompt_id], indent=2))
                break
        except Exception as e:
            pass
        time.sleep(2)
