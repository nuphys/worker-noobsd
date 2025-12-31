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

> [!NOTE]  
> `image_url` and refiner-based workflows are **not supported** in this version. This worker uses a single base model only.

### LoRA Support

This worker supports loading and applying LoRA (Low-Rank Adaptation) models to customize the output. LoRAs are optional and completely backward compatible—existing requests without LoRAs work exactly as before.

#### LoRA Configuration

The `loras` parameter accepts a list of LoRA configurations. Each LoRA can be specified as:

1. **String format** (uses default scale of 1.0):
   ```json
   "loras": [
     "https://example.com/my-lora.safetensors",
     "username/repo-name"
   ]
   ```

2. **Object format** (allows custom scale):
   ```json
   "loras": [
     {
       "path": "https://example.com/my-lora.safetensors",
       "scale": 0.8
     },
     {
       "path": "username/repo-name",
       "scale": 1.2
     }
   ]
   ```

#### LoRA Sources

LoRAs can be loaded from:
- **Direct URLs**: Full URL to a `.safetensors` or `.pt` file
- **HuggingFace repos**: Format `username/repo-name` (automatically downloads `pytorch_lora_weights.safetensors`)
- **Cached filenames**: Previously downloaded LoRA files stored in `/models/loras/`

#### Caching & Network Volume

LoRA files are automatically cached in `/models/loras/` on the network volume, ensuring:
- LoRAs persist across worker restarts
- Subsequent requests using the same LoRA load instantly from cache
- No redundant downloads

#### Scale Parameter

The `scale` parameter controls the strength of the LoRA effect (default: 1.0):
- `0.0` = No effect
- `1.0` = Full LoRA strength (default)
- `> 1.0` = Enhanced effect
- `< 1.0` = Reduced effect

### Example Request

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
    "num_images": 1
  }
}
```

which is producing an output like this:

```json
{
  "delayTime": 11449,
  "executionTime": 6120,
  "id": "447f10b8-c745-4c3b-8fad-b1d4ebb7a65b-e1",
  "output": {
    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU...",
    "images": [
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU..."
    ],
    "seed": 42
  },
  "status": "COMPLETED",
  "workerId": "462u6mrq9s28h6"
}
```

and when you convert the base64-encoded image into an actual image, it looks like this:

<img src="https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-sdxl_output_1-AedTpZlz1eIwIgAEShlod6syLo6Jq6.jpeg" alt="SDXL Generated Image: 'A majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed'" width="512" height="512">

### Example Request with LoRA

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
    "loras": [
      {
        "path": "https://civitai.com/api/download/models/123456",
        "scale": 0.85
      }
    ]
  }
}
```

The LoRA will be downloaded, cached to `/models/loras/`, and applied with the specified scale. Subsequent requests using the same LoRA URL will load instantly from cache.
