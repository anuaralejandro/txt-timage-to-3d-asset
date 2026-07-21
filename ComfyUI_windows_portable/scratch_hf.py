from huggingface_hub import HfApi

api = HfApi()

print("--- tencent/Hunyuan3D-2 ---")
files1 = api.list_repo_files("tencent/Hunyuan3D-2")
for f in files1:
    if f.endswith(".safetensors"):
        print(f)

print("--- tencent/Hunyuan3D-2mv ---")
files2 = api.list_repo_files("tencent/Hunyuan3D-2mv")
for f in files2:
    if f.endswith(".safetensors"):
        print(f)
