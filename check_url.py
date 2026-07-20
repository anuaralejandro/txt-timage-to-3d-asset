import urllib.request
import json
url = "https://api.github.com/repos/comfyanonymous/ComfyUI/releases/latest"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print("Release Name:", data.get('name'))
        for asset in data.get('assets', []):
            if 'windows_portable' in asset['name']:
                print("Found URL:", asset['browser_download_url'])
except Exception as e:
    print("Error:", e)
