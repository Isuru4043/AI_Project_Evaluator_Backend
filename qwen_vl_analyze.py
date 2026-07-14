## URL
## https://gayanshaminda655--qwen-vl-analyze-analyzer-analyze-api.modal.run

import modal
from fastapi import File, UploadFile, Form
from fastapi.responses import JSONResponse

app = modal.App("qwen-vl-analyze")

# Build the container image: CUDA base + transformers stack for Qwen2.5-VL
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.6.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.49.0",
        "accelerate",
        "qwen-vl-utils[decord]",
        "pillow",
        "huggingface_hub",
        "fastapi",
        "python-multipart",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Persistent volume so the model weights (~16GB for the 7B variant) are
# cached across runs instead of re-downloading on every cold start.
model_cache = modal.Volume.from_name("qwen-vl-cache", create_if_missing=True)
CACHE_DIR = "/cache"

# Default prompt used when Django doesn't send a custom one.
DEFAULT_PROMPT = (
    "Carefully read this image column by column, top to bottom. "
    "Extract ALL visible text exactly as written, including every "
    "numbered step, label, or callout — even ones in the middle or "
    "side of the image. Do not skip any numbered items, even if the "
    "layout is dense or multi-column. Double check your list against "
    "the image before finishing.\n\n"
    "Then, separately, describe any diagrams, charts, code structure, "
    "or figures shown, in plain language, including the overall flow "
    "or relationship between elements.\n\n"
    "Respond in this format:\n"
    "TEXT:\n<all extracted text, preserving numbering/order>\n\n"
    "DESCRIPTION:\n<description of diagrams/charts/figures, or 'None' "
    "if the slide is text-only>"
)


@app.cls(
    image=image,
    gpu="A10G",  # 24GB VRAM, comfortably fits 7B in bf16 with headroom to spare
    volumes={CACHE_DIR: model_cache},
    timeout=600,
    scaledown_window=120,  # keep container warm 2 min after last request
)
class Analyzer:
    @modal.enter()
    def load_model(self):
        """Runs once when the container starts, not on every request."""
        import os

        os.environ["HF_HOME"] = CACHE_DIR

        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.torch = torch
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)

    @modal.method()
    def analyze(self, image_bytes: bytes, prompt: str = DEFAULT_PROMPT) -> str:
        import io
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def analyze_api(self, image: UploadFile = File(...), prompt: str = Form(None)):
        """
        HTTP endpoint your Django backend calls whenever a changed slide
        frame is detected client-side.

        Example (Django, using `requests`):
            import requests
            with open("slide.jpg", "rb") as f:
                resp = requests.post(
                    "https://<your-modal-url>.modal.run",
                    files={"image": f},
                    data={"prompt": "optional custom prompt"},  # optional
                )
            result_text = resp.json()["result"]

        Example (curl):
            curl -X POST https://<your-url>.modal.run -F "image=@slide.jpg"
        """
        image_bytes = await image.read()
        used_prompt = prompt if prompt else DEFAULT_PROMPT
        try:
            result = self.analyze.local(image_bytes, prompt=used_prompt)
            return JSONResponse({"result": result, "filename": image.filename})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


@app.local_entrypoint()
def main(image_path: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    analyzer = Analyzer()
    result = analyzer.analyze.remote(image_bytes)
    print("\n--- VISION MODEL OUTPUT ---\n")
    print(result)