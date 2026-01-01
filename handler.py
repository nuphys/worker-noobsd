import os
import base64
import json
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
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
from PIL import PngImagePlugin
from huggingface_hub import hf_hub_download

# Compel 라이브러리
from compel import Compel, ReturnedEmbeddingsType

from schemas import INPUT_SCHEMA
from download_weights import download_lora, get_lora_cache_path

# 메모리 정리
torch.cuda.empty_cache()

# ==========================================
# [Face Detailer] Imports & Config
# ==========================================
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from ultralytics import YOLO
from diffusers import StableDiffusionXLImg2ImgPipeline

ADULTER_MODEL_DIR = "/runpod-volume/models/adulter"

class FaceDetailer:
    def __init__(self):
        self.detector = None
        self.i2i_pipe = None

    def ensure_model(self):
        os.makedirs(ADULTER_MODEL_DIR, exist_ok=True)
        model_path = os.path.join(ADULTER_MODEL_DIR, "face_yolov8n.pt")
        
        if not os.path.exists(model_path):
            print(f"📥 Downloading Face Detector to {model_path}...")
            try:
                hf_hub_download(
                    repo_id="Bingsu/adetailer",
                    filename="face_yolov8n.pt",
                    local_dir=ADULTER_MODEL_DIR,
                    local_dir_use_symlinks=False,
                    token=False 
                )
                print("✅ Face Detector downloaded.")
            except Exception as e:
                print(f"❌ Failed to download Face Detector: {e}")
                raise e
            
        return model_path

    def load_detector(self):
        if self.detector is None:
            model_path = self.ensure_model()
            print(f"Loading Face Detector from {model_path}")
            self.detector = YOLO(model_path)

    def load_i2i_pipeline(self, base_pipe):
        if self.i2i_pipe is None:
            print("Loading Img2Img Pipeline for Face Detailer...")
            self.i2i_pipe = StableDiffusionXLImg2ImgPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                text_encoder_2=base_pipe.text_encoder_2,
                tokenizer=base_pipe.tokenizer,
                tokenizer_2=base_pipe.tokenizer_2,
                unet=base_pipe.unet,
                scheduler=base_pipe.scheduler,
            )
            self.i2i_pipe.to("cuda")

    def detect_faces(self, image, conf=0.5):
        self.load_detector()
        # [수정] confidence 값 전달
        results = self.detector(image, conf=conf)
        boxes = []
        for result in results:
            boxes.extend(result.boxes.xyxy.cpu().numpy())
        return boxes

    def process(self, image, base_pipe, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, options):
        print("🕵️ Running Face Detailer...")
        
        # [수정] 모든 파라미터 추출 및 기본값 설정
        strength = options.get('strength', 0.4)
        padding = options.get('padding', 32)
        confidence = options.get('confidence', 0.5)
        guidance_scale = options.get('guidance_scale', 7.5)
        steps = options.get('num_inference_steps', 20)
        blur_sigma = options.get('blur_sigma', None)  # None이면 자동 계산
        resolution = options.get('resolution', 1024)  # SDXL 권장 해상도
        min_face_size = options.get('min_face_size', 64)

        # 얼굴 감지 실행
        boxes = self.detect_faces(image, conf=confidence)
        if len(boxes) == 0:
            print("No faces detected, skipping detailer.")
            return image

        print(f"Found {len(boxes)} faces. Starting enhancement...")
        
        self.load_i2i_pipeline(base_pipe)
        
        output_image = image.copy()
        w, h = output_image.size

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            
            # 패딩 적용
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face_w = x2 - x1
            face_h = y2 - y1
            
            # 너무 작은 얼굴 무시
            if face_w < min_face_size or face_h < min_face_size:
                print(f"Skipping small face: {face_w}x{face_h}")
                continue

            # 얼굴 잘라내기
            face_crop = output_image.crop((x1, y1, x2, y2))
            
            # [수정] 설정된 해상도로 리사이즈 (기본 1024)
            target_size = (resolution, resolution)
            face_crop_resized = face_crop.resize(target_size, Image.LANCZOS)
            
            # 인페인팅 실행
            with torch.inference_mode():
                detailed_face = self.i2i_pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    image=face_crop_resized,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            
            # 원본 크기로 복구
            detailed_face = detailed_face.resize((face_w, face_h), Image.LANCZOS)
            
            # 마스크 생성 및 블러 처리
            mask = Image.new('L', (face_w, face_h), 0) # 검은색 배경
            mask_draw = ImageDraw.Draw(mask)
            
            # 블러 값 결정 (사용자 입력 없으면 자동 계산)
            current_blur = blur_sigma if blur_sigma is not None else max(10, min(face_w, face_h) // 20)
            
            # [중요] 마스크를 꽉 채우지 않고, 블러 크기만큼 안쪽으로 들여서 그립니다 (Inset)
            # 이렇게 해야 가장자리가 검은색(투명)으로 자연스럽게 떨어집니다.
            mask_inset = int(current_blur * 1.5)  # 블러 강도의 1.5배만큼 안쪽으로 축소
            
            # 예외 처리: 얼굴이 너무 작아서 inset이 불가능한 경우 최소값 사용
            if mask_inset * 2 >= face_w or mask_inset * 2 >= face_h:
                mask_inset = max(1, min(face_w, face_h) // 10)
            
            # 축소된 흰색 사각형 그리기
            mask_draw.rectangle(
                [mask_inset, mask_inset, face_w - mask_inset, face_h - mask_inset], 
                fill=255
            )
            
            # 블러 적용
            mask = mask.filter(ImageFilter.GaussianBlur(current_blur))
            # -----------------------------------------------------------------------
            
            # 합성
            output_image.paste(detailed_face, (x1, y1), mask)
            
        print("Face Detailer complete.")
        return output_image
        
DETAIL_HANDLER = FaceDetailer()

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/noobai-xl-1.1.safetensors")

def pad_embeds(embeds1, embeds2):
    len1 = embeds1.shape[1]
    len2 = embeds2.shape[1]
    if len1 == len2: return embeds1, embeds2
    if len1 > len2:
        diff = len1 - len2
        padding = torch.zeros((embeds2.shape[0], diff, embeds2.shape[2]), dtype=embeds2.dtype, device=embeds2.device)
        embeds2 = torch.cat([embeds2, padding], dim=1)
    else:
        diff = len2 - len1
        padding = torch.zeros((embeds1.shape[0], diff, embeds1.shape[2]), dtype=embeds1.dtype, device=embeds1.device)
        embeds1 = torch.cat([embeds1, padding], dim=1)
    return embeds1, embeds2

class ModelHandler:
    def __init__(self):
        self.base = None
        self.compel = None
        self.load_models()

    def load_base(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=False,
        )
        print(f"Loading NoobAI XL 1.1 from {MODEL_PATH}")
        base_pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATH,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        )
        base_pipe.to("cuda")
        base_pipe.enable_xformers_memory_efficient_attention()
        
        self.compel = Compel(
            tokenizer=[base_pipe.tokenizer, base_pipe.tokenizer_2],
            text_encoder=[base_pipe.text_encoder, base_pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False 
        )
        return base_pipe

    def load_models(self):
        self.base = self.load_base()

MODELS = ModelHandler()

def _load_loras(pipeline, loras_config):
    if not loras_config: return None, None
    lora_paths, lora_scales = [], []
    for lora_config in loras_config:
        if isinstance(lora_config, str):
            lora_source, lora_name = lora_config, None
        elif isinstance(lora_config, dict):
            lora_source = lora_config.get('path') or lora_config.get('url') or lora_config.get('name')
            lora_name = lora_config.get('name')
        else: continue
        if not lora_source: continue
        try:
            lora_path = download_lora(lora_source, custom_name=lora_name)
            if lora_path:
                lora_paths.append(lora_path)
                lora_scales.append(lora_config.get('scale', 1.0) if isinstance(lora_config, dict) else 1.0)
                print(f"Loaded LoRA: {lora_path}")
        except Exception as e:
            print(f"Error processing LoRA {lora_source}: {e}")
            continue
    return (lora_paths, lora_scales) if lora_paths else (None, None)

def _apply_loras_to_pipeline(pipeline, lora_paths, lora_scales):
    if not lora_paths: return
    try:
        for idx, (lora_path, lora_scale) in enumerate(zip(lora_paths, lora_scales)):
            pipeline.load_lora_weights(lora_path, adapter_name=f"lora_{idx}")
        if len(lora_paths) == 1:
            pipeline.set_adapters(["lora_0"], adapter_weights=[lora_scales[0]])
        else:
            pipeline.set_adapters([f"lora_{i}" for i in range(len(lora_paths))], adapter_weights=lora_scales)
    except Exception as e:
        raise RuntimeError(f"Failed to apply LoRAs: {e}")

def _unload_loras_from_pipeline(pipeline):
    try: pipeline.unload_lora_weights()
    except Exception: pass

def _save_and_upload_images(images, job_id, job_input):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    metadata = PngImagePlugin.PngInfo()
    try: metadata.add_text("parameters", json.dumps(job_input, default=str, ensure_ascii=False))
    except Exception: pass

    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path, pnginfo=metadata)
        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_urls.append(rp_upload.upload_image(job_id, image_path))
        else:
            with open(image_path, "rb") as image_file:
                image_urls.append(f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}")
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
    job_input = job["input"]
    try: validated_input = validate(job_input, INPUT_SCHEMA)
    except Exception as err: return {"error": f"Validation Error: {err}"}
    if "errors" in validated_input: return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]
    if job_input["seed"] is None: job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input["seed"])
    MODELS.base.scheduler = make_scheduler(job_input["scheduler"], MODELS.base.scheduler.config)

    # LoRA 적용
    loras_config = job_input.get("loras")
    lora_paths = None
    if loras_config:
        try:
            lora_paths, lora_scales = _load_loras(MODELS.base, loras_config)
            if lora_paths: _apply_loras_to_pipeline(MODELS.base, lora_paths, lora_scales)
        except Exception as e: return {"error": f"LoRA Error: {str(e)}", "refresh_worker": False}

    try:
        # 1. Compel 임베딩 생성 (긴 프롬프트 처리)
        conditioning, pooled = MODELS.compel(job_input["prompt"])
        neg_conditioning, neg_pooled = MODELS.compel(job_input["negative_prompt"])

        # 2. 패딩 (길이 맞추기)
        conditioning, neg_conditioning = pad_embeds(conditioning, neg_conditioning)

        # 3. GPU 이동
        conditioning = conditioning.to("cuda")
        pooled = pooled.to("cuda")
        neg_conditioning = neg_conditioning.to("cuda")
        neg_pooled = neg_pooled.to("cuda")

        with torch.inference_mode():
            base_result = MODELS.base(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                height=job_input["height"],
                width=job_input["width"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=job_input["guidance_scale"],
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            )
            output = base_result.images

        # [수정] Face Detailer 실행 (버그 수정됨)
        face_detailer_config = job_input.get("face_detailer")
        if face_detailer_config:
            try:
                print("Processing Face Detailer...")
                detailed_images = []
                for img in output:
                    detailed_images.append(DETAIL_HANDLER.process(
                        img, 
                        MODELS.base, 
                        prompt_embeds=conditioning,           # 긴 프롬프트 임베딩 전달
                        negative_prompt_embeds=neg_conditioning,
                        pooled_prompt_embeds=pooled,
                        negative_pooled_prompt_embeds=neg_pooled, # [수정 완료] 올바른 변수명 전달
                        options=face_detailer_config
                    ))
                output = detailed_images
            except Exception as e: 
                print(f"Face Detailer Error: {e}")
                import traceback
                traceback.print_exc()
                # 에러 나도 원본 이미지는 살림

    except Exception as err:
        import traceback
        traceback.print_exc()
        return {"error": f"Generation Error: {err}", "refresh_worker": True}
    finally:
        # [중요] 모든 생성이 끝난 뒤에 LoRA 해제
        if loras_config and lora_paths: _unload_loras_from_pipeline(MODELS.base)

    image_urls = _save_and_upload_images(output, job["id"], job_input)
    return {"images": image_urls, "image_url": image_urls[0], "seed": job_input["seed"]}

if __name__ == "__main__":
    runpod.serverless.start({"handler": generate_image})