import json
import urllib.request
import time
import sys

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

with open("ComfyUI_windows_portable/ComfyUI/user/api_hunyuan3d_part_custom.json", "r") as f:
    prompt = json.load(f)

print("Submitting prompt to ComfyUI (waiting for boot)...")
max_retries = 30
for i in range(max_retries):
    try:
        response = queue_prompt(prompt)
        print("Response:", response)
        break
    except urllib.error.URLError as e:
        if i == max_retries - 1:
            print(f"Failed to connect after {max_retries} attempts.")
            sys.exit(1)
        time.sleep(3)
        print(f"Waiting for server... ({i+1}/{max_retries})")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

try:
    
    # Check history to see if it finished
    prompt_id = response['prompt_id']
    while True:
        history = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/history").read())
        if prompt_id in history:
            print("Finished!")
            print(json.dumps(history[prompt_id], indent=2))
            break
        time.sleep(2)

except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.read().decode('utf-8')}")
except Exception as e:
    print(f"Error: {e}")
