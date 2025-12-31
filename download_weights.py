import os
import torch
from diffusers import AutoencoderKL

# Model configuration constants
CHECKPOINT_PATH = "/models/noobai-xl-1.1.safetensors"
LORA_CACHE_DIR = "/models/loras"
CIVITAI_MODEL_ID = "1116447"
CIVITAI_DOWNLOAD_URL = f"https://civitai.com/api/download/models/{CIVITAI_MODEL_ID}?type=Model&format=SafeTensor&size=full&fp=bf16"
HF_FALLBACK_URL = "https://huggingface.co/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors?download=true"
DOWNLOAD_TIMEOUT = 300  # seconds


def fetch_pretrained_model(model_class, model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise


def download_file(url, destination, headers=None):
    """
    Download a file from a URL to a destination path.
    """
    import urllib.request
    
    print(f"Downloading from {url}")
    print(f"Destination: {destination}")
    
    request = urllib.request.Request(url, headers=headers or {})
    
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            total_size = response.headers.get('content-length')
            if total_size:
                total_size = int(total_size)
                print(f"Total size: {total_size / (1024**3):.2f} GB")
            
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            with open(destination, 'wb') as f:
                downloaded = 0
                chunk_size = 8192 * 16  # 128KB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress = (downloaded / total_size) * 100
                        print(f"Progress: {progress:.1f}% ({downloaded / (1024**3):.2f} GB)", end='\r')
            
            print(f"\nDownload complete: {destination}")
            return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_noobai_checkpoint():
    """
    Downloads the NoobAI XL 1.1 checkpoint file.
    Tries Civitai first (with optional token), falls back to HuggingFace.
    """
    # Skip if already exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint already exists at {CHECKPOINT_PATH}")
        return CHECKPOINT_PATH
    
    # Try Civitai first (preferred)
    civitai_token = os.environ.get("CIVITAI_API_TOKEN")
    
    headers = {}
    if civitai_token:
        headers["Authorization"] = f"Bearer {civitai_token}"
        print("Using Civitai API token for authenticated download")
    else:
        print("No CIVITAI_API_TOKEN found, attempting unauthenticated download")
    
    print("Attempting download from Civitai...")
    if download_file(CIVITAI_DOWNLOAD_URL, CHECKPOINT_PATH, headers):
        return CHECKPOINT_PATH
    
    # Fallback to HuggingFace
    print("Civitai download failed, trying HuggingFace fallback...")
    
    if download_file(HF_FALLBACK_URL, CHECKPOINT_PATH):
        return CHECKPOINT_PATH
    
    raise RuntimeError("Failed to download NoobAI XL 1.1 checkpoint from both sources")


def download_vae():
    """
    Downloads the SDXL VAE fix from HuggingFace.
    """
    vae = fetch_pretrained_model(
        AutoencoderKL, 
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    return vae


def get_lora_cache_path(lora_source):
    """
    Generate a cache path for a LoRA based on its source.
    
    Args:
        lora_source: URL, HuggingFace repo, or filename
        
    Returns:
        Full path to the cached LoRA file
    """
    import hashlib
    import re
    
    os.makedirs(LORA_CACHE_DIR, exist_ok=True)
    
    # If it's already a local path in the cache dir, return as-is
    if lora_source.startswith(LORA_CACHE_DIR):
        return lora_source
    
    # If it's just a filename (no path separators), treat as cached file
    if '/' not in lora_source and '\\' not in lora_source:
        return os.path.join(LORA_CACHE_DIR, lora_source)
    
    # Extract filename from URL or path
    if lora_source.startswith('http://') or lora_source.startswith('https://'):
        # For URLs, extract filename or create hash-based name
        url_parts = lora_source.split('?')[0]  # Remove query params
        filename = url_parts.split('/')[-1]
        
        # If no extension or generic name, use hash
        if not filename or '.' not in filename or filename in ['download', 'resolve']:
            url_hash = hashlib.md5(lora_source.encode()).hexdigest()[:12]
            filename = f"lora_{url_hash}.safetensors"
        
        # Ensure safe filename
        filename = re.sub(r'[^\w\-.]', '_', filename)
    else:
        # For HF repo format (org/repo) or other paths
        filename = lora_source.replace('/', '_').replace('\\', '_')
        if not filename.endswith(('.safetensors', '.pt', '.bin')):
            filename += '.safetensors'
    
    return os.path.join(LORA_CACHE_DIR, filename)


def download_lora(lora_source):
    """
    Download a LoRA file from a URL or HuggingFace repo to the cache directory.
    
    Args:
        lora_source: URL to LoRA file, HuggingFace repo reference, or local filename
        
    Returns:
        Path to the cached LoRA file, or None if download fails
    """
    cache_path = get_lora_cache_path(lora_source)
    
    # If already cached, return the path
    if os.path.exists(cache_path):
        print(f"LoRA already cached at {cache_path}")
        return cache_path
    
    print(f"Downloading LoRA from {lora_source}")
    
    # Handle HTTP(S) URLs
    if lora_source.startswith('http://') or lora_source.startswith('https://'):
        if download_file(lora_source, cache_path):
            print(f"LoRA downloaded successfully to {cache_path}")
            return cache_path
        else:
            print(f"Failed to download LoRA from {lora_source}")
            return None
    
    # Handle HuggingFace repo references
    # Format: "username/repo" or "username/repo/blob/main/filename.safetensors"
    if '/' in lora_source and not lora_source.startswith('/'):
        try:
            from huggingface_hub import hf_hub_download
            
            # Parse HF repo reference
            parts = lora_source.split('/')
            if len(parts) >= 2:
                repo_id = f"{parts[0]}/{parts[1]}"
                
                # Check if specific file is mentioned
                if len(parts) > 2 and 'blob' in parts:
                    # Format: username/repo/blob/main/filename.safetensors
                    filename = parts[-1]
                else:
                    # Assume default LoRA filename
                    filename = "pytorch_lora_weights.safetensors"
                
                print(f"Downloading from HuggingFace: {repo_id}/{filename}")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=LORA_CACHE_DIR,
                )
                
                # Copy or symlink to our cache path
                import shutil
                shutil.copy2(downloaded_path, cache_path)
                print(f"LoRA downloaded successfully to {cache_path}")
                return cache_path
        except ImportError:
            print("huggingface_hub not available, cannot download from HF")
        except Exception as e:
            print(f"Failed to download LoRA from HuggingFace: {e}")
            return None
    
    # If it's a local file reference that doesn't exist
    print(f"LoRA source not found: {lora_source}")
    return None


if __name__ == "__main__":
    print("Downloading NoobAI XL 1.1 checkpoint...")
    download_noobai_checkpoint()
    
    print("\nDownloading VAE...")
    download_vae()
    
    print("\nAll downloads complete!")
