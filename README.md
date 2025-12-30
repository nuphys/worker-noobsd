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

> [!NOTE]  
> `image_url` and refiner-based workflows are **not supported** in this version. This worker uses a single base model only.

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
