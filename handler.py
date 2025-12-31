import os
import base64
import json  # [추가] 메타데이터 변환용
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
)

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

# [추가] 이미지에 메타데이터를 심기 위해 필요
from PIL import PngImagePlugin

from schemas import INPUT_SCHEMA
from download_weights import download_lora, get_lora_cache_path

torch.cuda.empty_cache()

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/noobai-xl-1.1.safetensors")


class ModelHandler:
    def __init__(self):
        self.base = None
        self.load_models()

    def load_base(self):
        # Load VAE from cache using identifier
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=False,
        )
        
        # Changed: Use MODEL_PATH instead of CHECKPOINT_PATH
        print(f"Loading NoobAI XL 1.1 from {MODEL_PATH}")
        
        base_pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATH,  # This is the key change
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        ).to("cuda")
        
        # Enable memory optimizations
        base_pipe.enable_xformers_memory_efficient_attention()
        base_pipe.enable_model_cpu_offload()

        return base_pipe

    def load_models(self):
        self.base = self.load_base()


MODELS = ModelHandler()


def _load_loras(pipeline, loras_config):
    """
    Load and apply LoRAs to the pipeline.
    """
    if not loras_config:
        return None, None
    
    lora_paths = []
    lora_scales = []
    
    for lora_config in loras_config:
        if isinstance(lora_config, str):
            lora_source = lora_config
            lora_name = None
        elif isinstance(lora_config, dict):
            # [수정] path가 없으면 name을 path(파일명)로 간주하는 로직 추가
            lora_source = lora_config.get('path') or lora_config.get('url') or lora_config.get('name')
            lora_name = lora_config.get('name')
            lora_scale = lora_config.get('scale', 1.0)
            
            # 그래도 소스(파일명 또는 URL)가 없으면 에러
            if not lora_source:
                print(f"Warning: LoRA config missing 'path' or 'name': {lora_config}")
                continue
        else:
            print(f"Warning: Invalid LoRA config format: {lora_config}")
            continue
        
        try:
            # [수정] custom_name 인자 전달
            lora_path = download_lora(lora_source, custom_name=lora_name)
            
            if not lora_path:
                print(f"Error: Failed to prepare LoRA from {lora_source}")
                continue # 실패하면 건너뛰기
                
            lora_paths.append(lora_path)
            lora_scales.append(lora_scale)
            print(f"Loaded LoRA: {lora_path} with scale {lora_scale}")
            
        except Exception as e:
            print(f"Error processing LoRA {lora_source}: {e}")
            continue
    
    if not lora_paths:
        return None, None
    
    return lora_paths, lora_scales

def _apply_loras_to_pipeline(pipeline, lora_paths, lora_scales):
    """
    Apply LoRAs to the pipeline using load_lora_weights.
    """
    if not lora_paths:
        return
    
    try:
        # For multiple LoRAs, we need to load them sequentially
        # diffusers supports loading multiple LoRAs
        for idx, (lora_path, lora_scale) in enumerate(zip(lora_paths, lora_scales)):
            adapter_name = f"lora_{idx}"
            print(f"Loading LoRA adapter '{adapter_name}' from {lora_path}")
            
            # Load the LoRA weights
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            
        # Set the adapter scales
        if len(lora_paths) == 1:
            # Single LoRA - set scale directly
            pipeline.set_adapters(["lora_0"], adapter_weights=[lora_scales[0]])
        else:
            # Multiple LoRAs - set all scales
            adapter_names = [f"lora_{i}" for i in range(len(lora_paths))]
            pipeline.set_adapters(adapter_names, adapter_weights=lora_scales)
            
        print(f"Applied {len(lora_paths)} LoRA(s) to pipeline")
        
    except Exception as e:
        print(f"Error applying LoRAs: {e}")
        raise RuntimeError(f"Failed to apply LoRAs: {e}")


def _unload_loras_from_pipeline(pipeline):
    """
    Unload all LoRAs from the pipeline to restore original state.
    """
    try:
        # Unload all LoRA adapters
        pipeline.unload_lora_weights()
        print("Unloaded all LoRAs from pipeline")
    except Exception as e:
        print(f"Warning: Error unloading LoRAs: {e}")


# [변경] job_input 인자 추가
def _save_and_upload_images(images, job_id, job_input):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []

    # [추가] 메타데이터 생성 로직
    # job_input 전체를 JSON 문자열로 변환하여 'parameters' 태그에 저장합니다.
    metadata = PngImagePlugin.PngInfo()
    try:
        # ensure_ascii=False를 쓰면 한글 프롬프트 등도 깨지지 않고 저장됩니다.
        metadata.add_text("parameters", json.dumps(job_input, default=str, ensure_ascii=False))
    except Exception as e:
        print(f"Warning: Failed to create metadata: {e}")

    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        
        # [변경] pnginfo 파라미터 추가하여 저장
        image.save(image_path, pnginfo=metadata)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "DPMSolverSinglestep": DPMSolverSinglestepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using your Model
    """
    # -------------------------------------------------------------------------
    # 🐞 DEBUG LOGGING
    # -------------------------------------------------------------------------
    import pprint

    # Log the exact structure RunPod delivers so we can see every nesting level.
    print("[generate_image] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)

    # -------------------------------------------------------------------------
    # Original (strict) behaviour – assume the expected single wrapper exists.
    # -------------------------------------------------------------------------
    job_input = job["input"]

    print("[generate_image] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)

    # Input validation
    try:
        validated_input = validate(job_input, INPUT_SCHEMA)
    except Exception as err:
        import traceback

        print("[generate_image] validate(...) raised an exception:", err, flush=True)
        traceback.print_exc()
        # Re-raise so RunPod registers the failure (but logs are now visible).
        raise

    print("[generate_image] validate(...) returned:")
    try:
        print(json.dumps(validated_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(validated_input, depth=4, compact=False)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    # Create generator with proper device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(job_input["seed"])

    MODELS.base.scheduler = make_scheduler(
        job_input["scheduler"], MODELS.base.scheduler.config
    )

    # Load and apply LoRAs if specified
    loras_config = job_input.get("loras")
    lora_paths = None
    lora_scales = None
    
    if loras_config:
        print(f"LoRA configuration provided: {loras_config}")
        try:
            lora_paths, lora_scales = _load_loras(MODELS.base, loras_config)
            if lora_paths and lora_scales:
                _apply_loras_to_pipeline(MODELS.base, lora_paths, lora_scales)
        except Exception as e:
            print(f"Error loading LoRAs: {e}")
            return {
                "error": f"Failed to load LoRAs: {str(e)}",
                "refresh_worker": False,
            }

    try:
        # Generate image using base pipeline only (no refiner)
        with torch.inference_mode():
            base_result = MODELS.base(
                prompt=job_input["prompt"],
                negative_prompt=job_input["negative_prompt"],
                height=job_input["height"],
                width=job_input["width"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=job_input["guidance_scale"],
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            )
            output = base_result.images
    except RuntimeError as err:
        print(f"[ERROR] RuntimeError in generation pipeline: {err}", flush=True)
        return {
            "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
            "refresh_worker": True,
        }
    except Exception as err:
        print(f"[ERROR] Unexpected error in generation pipeline: {err}", flush=True)
        return {
            "error": f"Unexpected error: {err}",
            "refresh_worker": True,
        }
    finally:
        # Always unload LoRAs after generation to restore pipeline state
        if loras_config and lora_paths:
            _unload_loras_from_pipeline(MODELS.base)

    # [변경] output과 job_id 외에 job_input도 함께 전달
    image_urls = _save_and_upload_images(output, job["id"], job_input)

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }

    return results


runpod.serverless.start({"handler": generate_image})
