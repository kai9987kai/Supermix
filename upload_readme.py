from huggingface_hub import HfApi
import os

def main():
    api = HfApi()
    repo_id = "Kai9987kai/supermix-titan-dreamer-v43"
    readme_path = r"C:\Users\kai99\.gemini\antigravity\brain\df12b79b-fe06-4e1e-8d1f-6d55a5ec2665\MODEL_CARD_V43_TITAN_DREAMER.md"
    
    if os.path.exists(readme_path):
        print(f"Uploading README to {repo_id}...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id
        )
        print("Model card uploaded successfully!")
    else:
        print("Model card file not found.")

if __name__ == "__main__":
    main()
