from huggingface_hub import HfApi
import os

def main():
    api = HfApi()
    repo_id = "Kai9987kai/supermix-titan-dreamer-v43"
    print(f"Creating repo {repo_id}...")
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    
    files_to_upload = [
        "source/champion_model_chat_titan_dreamer_v43_ft.pth",
        "source/chat_model_meta_titan_dreamer_v43.json"
    ]
    
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id
            )
        else:
            print(f"File not found: {file_path}")
            
    print(f"Upload complete! View at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
