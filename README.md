![NoobAI XL Worker Banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-sdxl_banner-c7nsJLBOGHnmsxcshN7kSgALHYawnW.jpeg)

---

Run [NoobAI XL 1.1](https://civitai.com/models/833294?modelVersionId=1116447) as a serverless endpoint to generate images.

NoobAI XL 1.1 is an SDXL-based anime model trained on high-quality datasets, optimized for generating detailed anime-style artwork.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-sdxl)](https://www.runpod.io/console/hub/runpod-workers/worker-sdxl)

---

## Model Information

This worker uses **NoobAI XL 1.1** as a single base model (no refiner). The model is downloaded at build time from Civitai.

### Recommended Settings for NoobAI XL 1.1

- **Guidance Scale (CFG):** 5-6 (default: 5.5)
- **Inference Steps:** 25-30 (default: 28)
- **Sampler:** Euler Ancestral (K_EULER_ANCESTRAL)
- **Resolution:** 1024x1024 or similar SDXL resolutions

### Build-Time Configuration

To download from Civitai (optional for authenticated access):
```bash
docker build --build-arg CIVITAI_API_TOKEN=your_token_here -t worker-noobai .
```

If no token is provided, the build will attempt an unauthenticated download. If Civitai fails, it falls back to Hugging Face.

> **⚠️ Security Note:** Never commit your API token to the repository. Always pass it as a build argument or environment variable.

---

## Usage

The worker accepts the following input parameters:

| Parameter             | Type    | Default              | Required  | Description                                                                                        |
| :-------------------- | :------ | :------------------- | :-------- | :------------------------------------------------------------------------------------------------- |
| `prompt`              | `str`   | `None`               | **Yes**   | The main text prompt describing the desired image.                                                 |
| `negative_prompt`     | `str`   | `None`               | No        | Text prompt specifying concepts to exclude from the image                                          |
| `height`              | `int`   | `1024`               | No        | The height of the generated image in pixels                                                        |
| `width`               | `int`   | `1024`               | No        | The width of the generated image in pixels                                                         |
| `seed`                | `int`   | `None`               | No        | Random seed for reproducibility. If `None`, a random seed is generated                             |
| `scheduler`           | `str`   | `'K_EULER_ANCESTRAL'`| No        | The noise scheduler to use. Options: `PNDM`, `KLMS`, `DDIM`, `K_EULER`, `K_EULER_ANCESTRAL`, `DPMSolverMultistep`, `DPMSolverSinglestep` |
| `num_inference_steps` | `int`   | `28`                 | No        | Number of denoising steps                                                                          |
| `guidance_scale`      | `float` | `5.5`                | No        | Classifier-Free Guidance scale. Higher values lead to images closer to the prompt                  |
| `num_images`          | `int`   | `1`                  | No        | Number of images to generate per prompt (Constraint: must be 1 or 2)                               |
| `loras`               | `list`  | `None`               | No        | Optional list of LoRA configurations to apply. See LoRA section below for details                  |
| `face_detailer`       | `dict`  | `None`               | No        | Configuration for Face Detailer (ADetailer). See dedicated section below.                          |

> [!NOTE]  
> `image_url` and refiner-based workflows are **not supported** in this version. This worker uses a single base model only.

---

### Face Detailer Configuration

You can improve face quality and fix distortions using the built-in Face Detailer. It uses a YOLO model to detect faces and re-generates them with high resolution.

To avoid rectangular artifacts (seams) around the face, we recommend using a high `padding` value and the `blur_sigma` option.

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `strength` | `float` | `0.4` | Denoising strength. Lower is closer to original, higher is more creative. (Rec: 0.35-0.4) |
| `padding` | `int` | `32` | Pixels to add around the detected face. **Increase to 64-96 to avoid boxy borders.** |
| `confidence` | `float` | `0.5` | Minimum confidence threshold for face detection. |
| `guidance_scale` | `float` | `7.5` | CFG scale for the detailer. **Lowering to 4.0-5.0 helps blend with the background.** |
| `blur_sigma` | `int` | `Auto` | Blur radius for the mask edge. If `null`, it's calculated automatically. Set to ~20 for smoother blending. |
| `resolution` | `int` | `1024` | Internal processing resolution for the face crop. |
| `num_inference_steps`| `int` | `20` | Number of inference steps for the detailer. |

#### Example Request with Face Detailer

```json
{
  "input": {
    "prompt": "1girl, solo, close up, detailed face...",
    "face_detailer": {
      "strength": 0.35,
      "padding": 72,
      "confidence": 0.5,
      "guidance_scale": 4.5,
      "blur_sigma": 20,
      "resolution": 1024
    }
  }
}
```

---

### LoRA Support & Caching

This worker supports advanced LoRA loading with server-side caching and aliasing. You can name your LoRAs to reuse them later without re-downloading.

#### LoRA Loading Logic

The worker operates in 3 modes based on the presence of `name` and `path` in the input:

| Mode | Input Data | Behavior |
| :--- | :--- | :--- |
| **1. Register / Cache**<br>(New Download) | **Name (O) + URL (O)** | 1. Check if a file with `name` exists on the server.<br>2. **If exists:** Use the cached file (skip download).<br>3. **If missing:** Download from `path` and save as `name.safetensors`. |
| **2. Reuse**<br>(Load by Name) | **Name (O) + URL (X)** | 1. Check if a file with `name` exists.<br>2. **If exists:** Load immediately.<br>3. **If missing:** **Error** (File not found). |
| **3. Anonymous**<br>(One-time Use) | **Name (X) + URL (O)** | 1. Generate a filename based on the URL hash.<br>2. Download (or use hash-cache) and apply.<br>*(Legacy behavior)* |

#### JSON Payload Examples

**Case A: Registering/Downloading a LoRA with a Name**
*Use this when you want to download a LoRA and save it as "my_style" for future use.*
```json
"loras": [
  {
    "path": "https://civitai.com/api/download/models/12345",
    "name": "my_style",
    "scale": 0.8
  }
]
```

**Case B: Reusing a Saved LoRA**
*Use this to load the "my_style" LoRA you downloaded previously. No URL needed.*
```json
"loras": [
  {
    "name": "my_style",
    "scale": 1.0
  }
]
```

**Case C: Direct URL (Anonymous)**
*Use this for quick testing without naming the file.*
```json
"loras": [
  {
    "path": "https://civitai.com/api/download/models/12345",
    "scale": 0.8
  }
]
```

---

### Full Example Request

```json
{
  "input": {
    "prompt": "A majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed",
    "negative_prompt": "blurry, low quality, deformed, ugly, text, watermark, signature",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 5.5,
    "seed": 42,
    "scheduler": "K_EULER_ANCESTRAL",
    "num_images": 1,
    "face_detailer": {
        "strength": 0.35,
        "padding": 64,
        "blur_sigma": 15
    },
    "loras": [
      {
        "path": "https://civitai.com/api/download/models/123456",
        "name": "steampunk_v1",
        "scale": 0.85
      }
    ]
  }
}
```
