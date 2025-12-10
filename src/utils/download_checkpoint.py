import argparse
from huggingface_hub import hf_hub_download
import os

def download_checkpoint(repo_id, filename, local_dir="checkpoints", token=None):
    """
    Downloads a specific file from a Hugging Face repository.
    
    Args:
        repo_id (str): The HF repo ID (e.g., "username/repo-name").
        filename (str): The filename to download (e.g., "model.pth").
        local_dir (str): Directory to save the file.
        token (str): HF API token (optional, prioritizes HF_TOKEN env var).
    """
    print(f"⬇️  Downloading '{filename}' from '{repo_id}'...")
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Download the actual file, not a symlink
            token=token
        )
        print(f"✅ Successfully downloaded to: {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model checkpoint from Hugging Face")
    parser.add_argument("repo_id", help="Hugging Face Repo ID (e.g., 'TarunNagarajan/early-exit')")
    parser.add_argument("filename", help="Filename to download (e.g., 'model.pth')")
    parser.add_argument("--dir", default="checkpoints", help="Local directory to save to")
    parser.add_argument("--token", help="Hugging Face Token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    download_checkpoint(args.repo_id, args.filename, args.dir, args.token)
