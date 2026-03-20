from huggingface_hub import HfApi
import os

api = HfApi()

model_path = "outputs/models/vgg16/final_VGG16.keras"

# Check file exists locally
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"File found: {model_path}")
    print(f"File size: {size_mb:.1f} MB")
else:
    print(f"ERROR: File not found at {model_path}")
    exit()

print("Starting upload...")

url = api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="outputs/models/vgg16/final_vgg16.keras",
    repo_id="rakib-ahmed/rice-disease-classifier",
    repo_type="space"
)

print(f"Upload complete!")
print(f"URL: {url}")