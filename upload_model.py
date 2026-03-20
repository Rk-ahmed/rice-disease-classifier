from huggingface_hub import HfApi

api = HfApi()

print("Uploading model file... this may take a few minutes...")

api.upload_file(
    path_or_fileobj="outputs/models/vgg16/final_VGG16.keras",
    path_in_repo="outputs/models/vgg16/final_VGG16.keras",
    repo_id="rakib-ahmed/rice-disease-classifier",
    repo_type="space"
)

print("Model upload complete!")