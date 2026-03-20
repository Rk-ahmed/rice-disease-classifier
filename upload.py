from huggingface_hub import HfApi

api = HfApi()

print("Uploading files to Hugging Face...")

api.upload_folder(
    folder_path=".",
    repo_id="rakib-ahmed/rice-disease-classifier",
    repo_type="space",
    ignore_patterns=[
        "venv/*",
        "__pycache__/*",
        "*.pyc",
        "outputs/models/*",
        ".git/*",
        "data/*"
    ]
)

print("Upload complete!")