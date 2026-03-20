from huggingface_hub import HfApi

api = HfApi()

print("Uploading model with correct filename...")

api.upload_file(
    path_or_fileobj="outputs/models/vgg16/final_VGG16.keras",
    path_in_repo="outputs/models/vgg16/final_vgg16.keras",
    repo_id="rakib-ahmed/rice-disease-classifier",
    repo_type="space"
)

print("Done!")