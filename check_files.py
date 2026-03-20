from huggingface_hub import HfApi

api = HfApi()

files = api.list_repo_files(
    repo_id="rakib-ahmed/rice-disease-classifier",
    repo_type="space"
)

for f in files:
    print(f)