## URL
## https://gayanshaminda655--canary-qwen-transcribe-transcriber-tra-f02024.modal.run

import modal
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse

app = modal.App("canary-qwen-transcribe")

# Build the container image: CUDA base + NeMo toolkit (this is the heavy part)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.6.0",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("sacrebleu")
    .pip_install(
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"
    )
    .pip_install("peft")
    .pip_install("huggingface_hub")
    .pip_install("fastapi", "python-multipart")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Persistent volume so the ~5GB model weights are cached across runs
# instead of re-downloading every time the container cold-starts.
model_cache = modal.Volume.from_name("canary-qwen-cache", create_if_missing=True)
CACHE_DIR = "/cache"


@app.cls(
    image=image,
    gpu="L4",
    volumes={CACHE_DIR: model_cache},
    timeout=600,
    scaledown_window=120,  # keep container warm 2 min after last request
)
class Transcriber:
    @modal.enter()
    def load_model(self):
        """Runs once when the container starts, not on every request."""
        import os

        os.environ["HF_HOME"] = CACHE_DIR

        import torch
        from nemo.collections.speechlm2.models import SALM

        self.torch = torch
        self.model = SALM.from_pretrained("nvidia/canary-qwen-2.5b").bfloat16().eval().to("cuda")

    @modal.method()
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        import subprocess
        import tempfile

        # Write incoming bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix="_" + filename, delete=False) as f:
            f.write(audio_bytes)
            raw_path = f.name

        # Normalize to 16kHz mono WAV (required input format)
        wav_path = raw_path + "_16k.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True,
        )

        answer_ids = self.model.generate(
            prompts=[
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {self.model.audio_locator_tag}",
                        "audio": [wav_path],
                    }
                ]
            ],
            max_new_tokens=1024,
        )
        transcript = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
        return transcript

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def transcribe_api(self, audio: UploadFile = File(...)):
        """
        HTTP endpoint your Django backend (or anything else) can call.

        Example (Django, using `requests`):
            import requests
            with open("clip.wav", "rb") as f:
                resp = requests.post(
                    "https://<your-modal-url>.modal.run",
                    files={"audio": f},
                )
            transcript = resp.json()["transcript"]

        Example (curl):
            curl -X POST https://<your-url>.modal.run -F "audio=@clip.wav"
        """
        audio_bytes = await audio.read()
        try:
            transcript = self.transcribe.local(audio_bytes, filename=audio.filename or "audio.wav")
            return JSONResponse({"transcript": transcript, "filename": audio.filename})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


@app.local_entrypoint()
def main(audio_path: str):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    import os

    transcriber = Transcriber()
    result = transcriber.transcribe.remote(audio_bytes, filename=os.path.basename(audio_path))
    print("\n--- TRANSCRIPT ---\n")
    print(result)