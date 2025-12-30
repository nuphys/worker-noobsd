import os
import torch
from diffusers import AutoencoderKL

# Model configuration constants
CHECKPOINT_PATH = "/models/noobai-xl-1.1.safetensors"
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


if __name__ == "__main__":
    print("Downloading NoobAI XL 1.1 checkpoint...")
    download_noobai_checkpoint()
    
    print("\nDownloading VAE...")
    download_vae()
    
    print("\nAll downloads complete!")
