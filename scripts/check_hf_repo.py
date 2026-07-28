from huggingface_hub import HfApi

api = HfApi()

for repo_id in ["microsoft/TRELLIS-image-large", "microsoft/TRELLIS.2-4B", "JeffreyXiang/TRELLIS-image-large"]:
    try:
        info = api.repo_info(repo_id)
        print(f"✅ REPO EXISTS: {repo_id}")
    except Exception as e:
        print(f"❌ REPO NOT FOUND ({repo_id}): {e}")
