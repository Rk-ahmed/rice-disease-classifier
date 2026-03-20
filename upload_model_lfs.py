from huggingface_hub import HfApi

api = HfApi()

print("Uploading with LFS support...")

api.upload_file(
    path_or_fileobj="outputs/models/vgg16/final_VGG16.keras",
    path_in_repo="outputs/models/vgg16/final_vgg16.keras",
    repo_id="rakib-ahmed/rice-disease-classifier",
    repo_type="space",
    commit_message="Add VGG16 model file"
)

print("Done!")

# Verify it exists
files = api.list_repo_files(
    repo_id="rakib-ahmed/rice-disease-classifier",
    repo_type="space"
)
print("\nAll files on HuggingFace:")
for f in files:
    if "model" in f or "outputs" in f:
        print(f"  FOUND: {f}")