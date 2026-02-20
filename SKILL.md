---
name: mlx
description: >
  Run local Apple MLX-based models for image generation (mflux — Flux, Z-Image, FLUX.2, Qwen),
  audio/TTS (mlx-audio), vision/multimodal analysis (mlx-vlm), and LLM text inference (mlx-lm).
  Handles environment discovery, silent dependency installation, model downloads, and execution.
  Use whenever the user asks to generate images, audio, text, or analyze images locally with MLX.
save_path: /Users/clawd/clawd/skills/mlx/SKILL.md
---

# MLX Local Model Runner

A skill for discovering, installing, and running Apple MLX-based models on the local machine.
MLX is Apple's machine learning framework optimized for Apple Silicon (M1/M2/M3/M4 chips).

---

## Step 0 — Environment Discovery (Always Run First)

Before doing anything else, **run this discovery script**. Never assume — always discover.
This output shapes every subsequent decision.

```python
import subprocess, sys, platform, json, os

info = {}
info["platform"] = platform.platform()
info["machine"] = platform.machine()

is_apple_silicon = platform.machine() == "arm64" and platform.system() == "Darwin"
info["apple_silicon"] = is_apple_silicon
info["mlx_viable"] = is_apple_silicon

try:
    v = platform.mac_ver()[0]
    info["macos_version"] = v
    parts = list(map(int, v.split(".")))
    info["macos_ok"] = parts[0] > 13 or (parts[0] == 13 and parts[1] >= 5)
except:
    info["macos_version"] = "unknown"
    info["macos_ok"] = False

try:
    mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
    info["ram_gb"] = round(int(mem) / 1e9, 1)
except:
    info["ram_gb"] = "unknown"

try:
    st = os.statvfs(os.path.expanduser("~"))
    info["free_disk_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
except:
    info["free_disk_gb"] = "unknown"

packages_to_check = ["mlx", "mflux", "mlx_audio", "mlx_vlm", "mlx_lm",
                     "huggingface_hub", "numpy", "pillow", "soundfile"]
installed = {}
for pkg in packages_to_check:
    try:
        mod = __import__(pkg)
        installed[pkg] = getattr(mod, "__version__", "installed")
    except ImportError:
        installed[pkg] = None
info["installed_packages"] = installed

# mflux cache (moved to ~/Library/Caches/mflux/ in v0.6+)
mflux_cache = os.path.expanduser("~/Library/Caches/mflux")
info["mflux_cache"] = mflux_cache
info["mflux_cache_exists"] = os.path.exists(mflux_cache)

hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
info["hf_cache"] = hf_cache
models_dir = os.path.join(hf_cache, "hub")
downloaded = []
if os.path.exists(models_dir):
    for item in os.listdir(models_dir):
        if item.startswith("models--"):
            downloaded.append(item.replace("models--", "").replace("--", "/"))
info["downloaded_models"] = downloaded

try:
    from huggingface_hub import HfApi
    info["hf_user"] = HfApi().whoami()["name"]
except:
    info["hf_user"] = None

# Check if uv is available (preferred mflux installer)
try:
    r = subprocess.check_output(["uv", "--version"], stderr=subprocess.DEVNULL).decode().strip()
    info["uv_available"] = r
except:
    info["uv_available"] = None

print(json.dumps(info, indent=2))
```

Parse this for:
- `apple_silicon` + `macos_ok` → MLX is viable; if false, stop and inform the user
- `ram_gb` → determines safe model sizes (see Step 2)
- `installed_packages` → what the silent-installer needs to handle
- `downloaded_models` → models already in HF cache (skip re-download)
- `hf_user` → whether HF auth is already set up
- `uv_available` → whether uv is present (preferred install path for mflux)

**If not Apple Silicon**: Inform the user MLX requires Apple Silicon Macs. Suggest cloud
alternatives like Replicate or fal.ai for image/audio generation.

---

## Step 1 — Map Request to Category

| Category | User says... | Package | Notes |
|----------|-------------|---------|-------|
| **Image generation** | "generate an image", "make art of X", "create a picture" | `mflux` | Multiple model families |
| **Audio / TTS** | "say this", "speak", "generate audio", "text to speech" | `mlx-audio` | Kokoro default |
| **Vision / VLM** | "describe this image", "what's in this photo", "analyze this" | `mlx-vlm` | Needs image input |
| **LLM / Chat** | "run a local model", "chat with Llama", "use a local LLM" | `mlx-lm` | Distributed-capable |
| **Music generation** | "generate music", "make a song" | `mlx-audio` (stable-audio) | Experimental |
| **Image → Vision pipeline** | "generate then describe", "make and analyze" | `mflux` + `mlx-vlm` | Chain both |
| **Image editing** | "edit this image", "change X in this photo" | `mflux` (Kontext/Qwen) | In-context editing |

---

## Step 2 — RAM-Based Model Selection

Use `ram_gb` from discovery to select appropriately. Never load a model that will OOM.

### Image Generation RAM Guide

```
8 GB   → Z-Image Turbo q8 (best choice) · flux-schnell q4 · FLUX.2 klein q8
16 GB  → Z-Image Turbo q8 · flux-schnell q8 · flux-dev q8 · FLUX.2 klein q8
32 GB  → flux-dev q8 (comfortable) · Qwen Image q6 · FLUX.2 dev q8
64 GB  → flux-dev full · Qwen Image q8 · FLUX.2 dev q6
96 GB+ → Any model at full precision
```

### LLM RAM Guide

```
8 GB   → Llama-3.2-3B-4bit · Phi-3.5-mini-4bit · Qwen2.5-3B-4bit
16 GB  → Llama-3.1-8B-4bit · Mistral-7B-4bit · Qwen2.5-7B-4bit
32 GB  → Llama-3.1-70B-4bit (tight) · Qwen2.5-32B-4bit
64 GB  → Llama-3.1-70B-4bit (comfortable) · DeepSeek-R1-32B-4bit
96 GB+ → DeepSeek-R1-70B-4bit · Llama-3.1-70B full precision
```

### Vision (VLM) RAM Guide

```
8 GB   → llava-1.5-7b-4bit
16 GB  → Qwen2-VL-7B-Instruct-4bit (best)
32 GB  → Qwen2-VL-7B (full) · InternVL2-8B
64 GB+ → Qwen2-VL-72B-Instruct-4bit
```

Always tell the user which model was selected and why (RAM-based).

---

## Step 3 — Silent Auto-Install

**Always install missing packages silently without asking the user.** Include this helper
at the top of every generated script:

```python
import subprocess, sys

def ensure(*packages):
    """Silently install any missing packages."""
    for pkg in packages:
        mod_name = pkg.replace("-", "_").split("[")[0]
        try:
            __import__(mod_name)
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q", "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
```

**mflux-specific**: The recommended install for mflux is via `uv tool` (if `uv` is available).
If `uv_available` is not None from discovery, use:

```python
import subprocess
# Install/upgrade mflux via uv tool (preferred)
subprocess.run(
    ["uv", "tool", "install", "--upgrade", "mflux", "--prerelease=allow"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
# Otherwise fall back to pip:
# ensure("mflux", "pillow")
```

Usage before imports:
```python
ensure("mflux", "pillow")             # image generation (fallback)
ensure("mlx-audio", "soundfile")      # audio / TTS
ensure("mlx-vlm")                     # vision
ensure("mlx-lm")                      # LLM inference
ensure("huggingface_hub")             # always useful
```

---

## Step 4 — Hugging Face Authentication

Check `hf_user` from discovery. If `None` and the requested model requires auth:

```python
from huggingface_hub import HfApi
try:
    print("Logged in as:", HfApi().whoami()["name"])
except:
    print("Not logged in — run: huggingface-cli login")
```

Guide the user:
```
1. Go to https://huggingface.co/settings/tokens
2. Create a free read token
3. Run: huggingface-cli login
   Or:  export HUGGING_FACE_HUB_TOKEN="hf_your_token"
```

**Models requiring HF login**: `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.2-dev`
**No login needed**: Z-Image, FLUX.2 klein (Apache 2.0), flux-schnell via mflux, Kokoro TTS, most mlx-community VLMs/LLMs

---

## Step 5 — Execution

### Image Generation (mflux ≥ 0.6)

> **API CHANGE from v0.4**: `Flux1.from_alias()` → `Flux1.from_name()`. `Config` object
> is removed — `num_inference_steps`, `height`, `width` are now direct kwargs on
> `generate_image()`. Always use the new API.

#### Model Families (choose based on RAM + use case)

| Model alias | CLI command | RAM (q8) | Steps | Best for |
|-------------|------------|----------|-------|---------|
| `schnell` | `mflux-generate --model schnell` | ~8 GB | 2–4 | Speed, drafts |
| `dev` | `mflux-generate --model dev` | ~8 GB | 20–25 | Quality |
| `krea-dev` | `mflux-generate --model krea-dev` | ~8 GB | 25 | Photorealism, avoids AI look |
| `z-image-turbo` | `mflux-generate-z-image-turbo` | ~6 GB | 9 | **Best all-rounder 2025** |
| `flux2-klein` | `mflux-generate --model flux2-klein` | ~5 GB | 4 | Fastest, real-time |
| `qwen` | `mflux-generate-qwen` | ~14 GB | 20 | Best prompt understanding |

**Recommendation**: Default to `z-image-turbo` unless user requests otherwise — it's the best
speed/quality balance as of 2025. Fall back to `schnell` on 8 GB RAM systems.

#### Python API (v0.6+ / latest)

```python
ensure("mflux", "pillow")

import os
from mflux.models.flux.variants.txt2img.flux import Flux1

out_dir = os.path.expanduser("~/Desktop/mlx-outputs")
os.makedirs(out_dir, exist_ok=True)

# RAM-based model selection (replace ram_gb with value from Step 0)
ram_gb = 16

if ram_gb >= 16:
    model_name = "z-image-turbo"   # best all-rounder
    steps = 9
    quantize = 8
elif ram_gb >= 8:
    model_name = "schnell"
    steps = 4
    quantize = 8
else:
    model_name = "schnell"
    steps = 4
    quantize = 4

flux = Flux1.from_name(
    model_name=model_name,
    quantize=quantize,
)

image = flux.generate_image(
    seed=42,
    prompt="A photorealistic cat sitting on a misty mountain at dawn",
    num_inference_steps=steps,
    height=1024,
    width=1024,
)

output_path = os.path.join(out_dir, "output.png")
image.save(path=output_path)
print(f"Saved: {output_path}")

import subprocess
subprocess.Popen(["open", output_path])
```

#### Z-Image Turbo (recommended default, 2025)

```python
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=8)
image = model.generate_image(
    prompt="A puffin standing on a cliff overlooking the ocean",
    seed=42,
    num_inference_steps=9,
    width=1280,
    height=500,
)
image.save(path="output.png")
```

CLI:
```bash
mflux-generate-z-image-turbo \
  --prompt "A puffin standing on a cliff" \
  --width 1280 --height 500 \
  --seed 42 --steps 9 -q 8
```

#### FLUX.2 Klein (fastest, Apache 2.0)

```bash
mflux-generate --model flux2-klein \
  --prompt "A serene Japanese garden at dawn" \
  --steps 4 --seed 42 -q 8
```

#### Flux-dev / Krea-dev (highest FLUX.1 quality)

```bash
# Standard dev
mflux-generate --model dev --prompt "your prompt" --steps 25 --seed 42 -q 8

# Krea-dev (photorealistic, avoids AI look)
mflux-generate --model krea-dev --prompt "A photo of a dog" --steps 25 --seed 2674888 -q 8
```

#### Image Editing (Kontext)

```bash
mflux-generate-kontext \
  --image-path original.jpg \
  --prompt "Change the sky to a stormy sunset" \
  --steps 25 --seed 42
```

#### Qwen Image (best prompt understanding, needs 14+ GB)

```bash
mflux-generate-qwen --prompt "Luxury food photograph" --steps 20 --seed 2 -q 6
```

#### Image-to-Image

```bash
mflux-generate --model dev \
  --prompt "Turn this into an oil painting" \
  --image-path input.jpg \
  --image-strength 0.6 \
  --steps 20 --seed 42 -q 8
```

**Quantization reference** (valid values: 3, 4, 5, 6, 8):
- `-q 4` → lowest RAM, fastest, slight quality loss
- `-q 6` → balanced for large models
- `-q 8` → best quality/RAM tradeoff (recommended default)
- no quantize → full precision, most RAM, best quality

**mflux cache location** (v0.6+): `~/Library/Caches/mflux/`
Set `MFLUX_CACHE_DIR` to override. HF model weights: `~/.cache/huggingface/`

---

### Audio / TTS (mlx-audio)

```python
ensure("mlx-audio", "soundfile", "numpy")

import os, soundfile as sf
from mlx_audio.tts.generate import generate_audio

out_dir = os.path.expanduser("~/Desktop/mlx-outputs")
os.makedirs(out_dir, exist_ok=True)

audio, sample_rate = generate_audio(
    text="Hello, this is a test of local MLX audio synthesis.",
    model="prince-canuma/Kokoro-82M",
    voice="af_heart",
    speed=1.0,
    lang_code="en-us",
)
output_path = os.path.join(out_dir, "output.wav")
sf.write(output_path, audio, sample_rate)
print(f"Saved: {output_path}")

import subprocess
subprocess.Popen(["afplay", output_path])
```

**TTS Models** (lightest → best quality):
| Model | RAM | Notes |
|-------|-----|-------|
| `prince-canuma/Kokoro-82M` | ~1 GB | Default — fast, great quality |
| `hexgrad/Kokoro-82M` | ~1 GB | Alternative Kokoro variant |
| `suno/bark-small` | ~2 GB | More expressive, slower |

**Kokoro voices**: `af` · `af_heart` · `af_bella` · `af_sarah` · `am_adam` · `bf_emma` · `bm_george`

---

### LLM / Text Inference (mlx-lm)

```python
ensure("mlx-lm")

from mlx_lm import load, generate

# RAM-based model selection (replace ram_gb from Step 0)
ram_gb = 16
if ram_gb >= 64:
    model_id = "mlx-community/Llama-3.1-70B-Instruct-4bit"
elif ram_gb >= 32:
    model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
elif ram_gb >= 16:
    model_id = "mlx-community/Llama-3.1-8B-Instruct-4bit"
elif ram_gb >= 8:
    model_id = "mlx-community/Llama-3.2-3B-Instruct-4bit"
else:
    model_id = "mlx-community/Phi-3.5-mini-instruct-4bit"

model, tokenizer = load(model_id)

# For chat-style prompts, apply the chat template
from mlx_lm.utils import make_kv_cache
messages = [{"role": "user", "content": "Explain quantum entanglement simply."}]
if hasattr(tokenizer, "apply_chat_template"):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
else:
    prompt = messages[0]["content"]

response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=512,
    verbose=True,
)
print(response)
```

**CLI shortcuts**:
```bash
# One-shot generation
mlx_lm.generate \
  --model mlx-community/Llama-3.1-8B-Instruct-4bit \
  --prompt "Explain quantum entanglement simply" \
  --max-tokens 512

# Interactive chat REPL
mlx_lm.chat --model mlx-community/Llama-3.1-8B-Instruct-4bit

# REST server (OpenAI-compatible endpoint on port 8080)
mlx_lm.server --model mlx-community/Llama-3.1-8B-Instruct-4bit --port 8080
```

**Recommended models by RAM**:
| RAM | Model | Notes |
|-----|-------|-------|
| 4–8 GB | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Lightweight, fast |
| 8–16 GB | `mlx-community/Llama-3.1-8B-Instruct-4bit` | Best 8B overall |
| 16 GB | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | Great all-rounder |
| 32 GB | `mlx-community/Qwen2.5-32B-Instruct-4bit` | Strong reasoning |
| 64+ GB | `mlx-community/Llama-3.1-70B-Instruct-4bit` | Near frontier quality |

---

### Vision / VLM (mlx-vlm)

```python
ensure("mlx-vlm", "pillow")

import os
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# RAM-based model selection
ram_gb = 16
if ram_gb >= 64:
    model_id = "mlx-community/Qwen2-VL-72B-Instruct-4bit"
elif ram_gb >= 16:
    model_id = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
else:
    model_id = "mlx-community/llava-1.5-7b-4bit"

model, processor = load(model_id)
config = load_config(model_id)

image_path = "path/to/image.jpg"
prompt = "Describe this image in detail."

formatted = apply_chat_template(processor, config, prompt, num_images=1)
response = generate(model, processor, image_path, formatted, verbose=False)
print(response)
```

**VLM Recommendations**:
| Model | RAM | Strength |
|-------|-----|---------|
| `mlx-community/llava-1.5-7b-4bit` | 8 GB | General vision Q&A |
| `mlx-community/Qwen2-VL-7B-Instruct-4bit` | 16 GB | Excellent OCR + detail |
| `mlx-community/Qwen2-VL-72B-Instruct-4bit` | 64 GB | Near-GPT4V quality |

---

### Combined: Image → Vision Pipeline

```python
ensure("mflux", "mlx-vlm", "pillow")

import os
from mflux.models.flux.variants.txt2img.flux import Flux1
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

out_dir = os.path.expanduser("~/Desktop/mlx-outputs")
os.makedirs(out_dir, exist_ok=True)
img_path = os.path.join(out_dir, "generated.png")

# 1. Generate with Z-Image Turbo (best all-rounder)
from mflux.models.z_image import ZImageTurbo
flux = ZImageTurbo(quantize=8)
image = flux.generate_image(
    prompt="A serene Japanese garden at dawn",
    seed=42,
    num_inference_steps=9,
    width=1024,
    height=1024,
)
image.save(path=img_path)
print(f"Image saved: {img_path}")

# 2. Analyze with VLM
model, processor = load("mlx-community/Qwen2-VL-7B-Instruct-4bit")
cfg = load_config("mlx-community/Qwen2-VL-7B-Instruct-4bit")
formatted = apply_chat_template(processor, cfg, "Describe this image in detail.", num_images=1)
description = generate(model, processor, img_path, formatted, verbose=False)
print(f"\nDescription:\n{description}")
```

---

## Step 6 — Output Handling

After every generation:
1. Print the output file path clearly
2. Auto-open/play on macOS:
   ```python
   import subprocess
   subprocess.Popen(["open", "/path/to/output.png"])   # images
   subprocess.Popen(["afplay", "/path/to/output.wav"]) # audio
   ```
3. If generation took >30 seconds, report timing so the user has expectations next time.

---

## Error Handling

| Error / Symptom | Cause | Fix |
|----------------|-------|-----|
| `machine != arm64` | Intel Mac / non-Mac | Inform user, MLX not supported |
| `RuntimeError: Out of memory` | Model too large for RAM | Higher quantization or smaller model |
| `SIGKILL` / process killed | macOS OOM killer | Use `--low-ram` flag or smaller model |
| `403 Forbidden` / `Repository not found` | HF auth required | `huggingface-cli login` |
| `ModuleNotFoundError` | Package missing | `ensure(...)` auto-handles |
| Very slow generation (>10 min) | No quantization | Add `-q 8` |
| Black or corrupted image | Stale mflux or MLX mismatch | `pip install -U mflux mlx` |
| VLM garbled output | Wrong prompt template | Use `apply_chat_template` from mlx_vlm |
| `Flux1.from_alias()` AttributeError | Old mflux API (< 0.6) | Upgrade: `pip install -U mflux` |
| `Config` import error | Old mflux API (< 0.6) | Same — use `from_name()` + direct kwargs |
| LLM hangs on first token | Model loading (normal) | Wait 30–60 sec on first load |
| `mlx_lm.server` not found | Old mlx-lm version | `pip install -U mlx-lm` |

---

## Extensibility — Adding New MLX Models

When a new MLX project appears, add it here:

1. **Find the PyPI package** — check the project's GitHub README
2. **Add `ensure("new-package")`** to Step 5 in the relevant section
3. **Add to the category table** in Step 1 if it's a new capability
4. **Add RAM requirements** to Step 2 RAM guides
5. **Add to the tracked projects table** below
6. **Add any new error patterns** to the Error Handling table

### Tracked MLX Projects

| Project | PyPI | HF Namespace | Status | Notes |
|---------|------|--------------|--------|-------|
| mflux (Flux, Z-Image, FLUX.2) | `mflux` | `black-forest-labs`, `Tongyi-MAI` | ✅ v0.16+ | Primary image gen |
| Audio / TTS | `mlx-audio` | `prince-canuma` | ✅ Stable | TTS, music |
| Vision / VLM | `mlx-vlm` | `mlx-community` | ✅ Stable | Image analysis |
| LLM inference | `mlx-lm` | `mlx-community` | ✅ Stable | Chat, server, distributed |
| MLX core | `mlx` | — | ✅ Stable | Dependency |
| Whisper STT | `mlx-whisper` | `mlx-community` | ✅ Stable | Speech-to-text |
| Stable Audio | via `mlx-audio` | `stabilityai` | 🧪 Experimental | Music generation |

---

## Quick Reference

```bash
# Image — Z-Image Turbo (best all-rounder, recommended default)
mflux-generate-z-image-turbo --prompt "your prompt" \
  --steps 9 --seed 42 -q 8 --width 1024 --height 1024

# Image — Flux-schnell (fast, good quality)
mflux-generate --model schnell --prompt "your prompt" --steps 4 -q 8

# Image — Flux-dev (highest Flux.1 quality)
mflux-generate --model dev --prompt "your prompt" --steps 25 -q 8

# Image — FLUX.2 Klein (fastest, Apache 2.0)
mflux-generate --model flux2-klein --prompt "your prompt" --steps 4 -q 8

# Image — Image editing with Kontext
mflux-generate-kontext --image-path photo.jpg --prompt "change the background to a beach" --steps 25

# Audio — TTS
python3 -c "
from mlx_audio.tts.generate import generate_audio; import soundfile as sf
audio, sr = generate_audio('Hello world', model='prince-canuma/Kokoro-82M', voice='af_heart')
sf.write('out.wav', audio, sr)
import subprocess; subprocess.Popen(['afplay', 'out.wav'])
"

# LLM — one-shot
mlx_lm.generate --model mlx-community/Llama-3.1-8B-Instruct-4bit --prompt "Hello"

# LLM — interactive chat
mlx_lm.chat --model mlx-community/Llama-3.1-8B-Instruct-4bit

# LLM — OpenAI-compatible server
mlx_lm.server --model mlx-community/Llama-3.1-8B-Instruct-4bit --port 8080

# Vision — describe an image
python3 -c "
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
m, p = load('mlx-community/Qwen2-VL-7B-Instruct-4bit')
c = load_config('mlx-community/Qwen2-VL-7B-Instruct-4bit')
f = apply_chat_template(p, c, 'Describe this image.', num_images=1)
print(generate(m, p, 'image.jpg', f, verbose=False))
"
```

---

## Decision Flowchart

```
User request
    │
    ├─ Not Apple Silicon  →  Stop. Suggest Replicate / fal.ai for cloud generation
    ├─ macOS < 13.5       →  Stop. Ask user to update macOS
    │
    ├─ Image generation
    │       RAM < 8 GB    →  z-image-turbo q4  OR  flux-schnell q4
    │       RAM 8-15 GB   →  z-image-turbo q8  (best all-rounder)
    │       RAM 16-31 GB  →  z-image-turbo q8  OR  flux-dev q8
    │       RAM 32+ GB    →  flux-dev q8  OR  Qwen Image q6
    │       Fastest?      →  FLUX.2 klein q8 (any RAM 8GB+)
    │       Photorealism? →  krea-dev q8
    │       Editing?      →  Kontext OR Qwen Image Edit
    │
    ├─ Audio / TTS
    │       Any RAM       →  Kokoro-82M (works everywhere)
    │       Expressive?   →  bark-small if RAM > 4 GB
    │
    ├─ LLM / text inference
    │       RAM 4-8 GB    →  Llama-3.2-3B-4bit
    │       RAM 8-16 GB   →  Llama-3.1-8B-4bit
    │       RAM 16-32 GB  →  Mistral-7B or Qwen2.5-14B-4bit
    │       RAM 32-64 GB  →  Qwen2.5-32B-4bit
    │       RAM 64+ GB    →  Llama-3.1-70B-4bit
    │       Distributed?  →  mlx.launch with ring backend (see Distributed section)
    │
    ├─ Vision / VLM
    │       RAM 8-15 GB   →  llava-1.5-7b-4bit
    │       RAM 16-31 GB  →  Qwen2-VL-7B-Instruct-4bit
    │       RAM 64+ GB    →  Qwen2-VL-72B-Instruct-4bit
    │
    ├─ Image + Vision pipeline  →  z-image-turbo → Qwen2-VL (see Step 5)
    │
    └─ New/unknown MLX model    →  Check mlx-community on HF, follow Extensibility guide
```

---

## Distributed Inference — LAN Cluster (LLM)

MLX supports distributed LLM inference across multiple Macs on a LAN using `mlx.launch`.
This pools RAM across machines to run models larger than any single Mac could hold.

> **Note:** Distributed inference currently applies to LLMs (mlx-lm). Image generation
> (mflux) and vision models (mlx-vlm) do not yet support multi-node sharding — use the
> single node with the most RAM for those.

### Backend Selection

| Backend | Transport | Requirements | Best For |
|---------|-----------|-------------|----------|
| `ring` | Ethernet/Wi-Fi TCP | SSH + same Python path | **LAN clusters — use this** |
| `jaccl` | Thunderbolt 5 RDMA | macOS 26.2+, TB5 cables | Directly-connected Macs |
| `mpi` | TCP via OpenMPI | OpenMPI installed | Legacy setups |

### Prerequisites

```bash
# Enable Remote Login on each node (System Settings → Sharing → Remote Login)
# Or via Terminal on each node:
sudo systemsetup -setremotelogin on

# Set up passwordless SSH from controller to each node
ssh-keygen -t ed25519 -C "mlx-cluster" -f ~/.ssh/id_mlx -N ""
for HOST in mac-mini-2.local mac-mini-3.local; do
    ssh-copy-id -i ~/.ssh/id_mlx.pub "$HOST"
done

# Verify (no password prompt)
ssh mac-mini-2.local "echo 'OK'"
```

### Cluster Health Check

```python
import subprocess, json

HOSTS = ["mac-mini-2.local", "mac-mini-3.local"]
PYTHON = "/usr/local/bin/python3"  # must be identical on ALL nodes

total_ram, all_ok = 0, True
for host in HOSTS:
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host,
         f"{PYTHON} -c \"import mlx, mlx_lm, platform, subprocess, json; "
         f"mem=int(subprocess.check_output(['sysctl','-n','hw.memsize']).decode()); "
         f"print(json.dumps({{'host':'{host}','mlx':mlx.__version__,"
         f"'mlx_lm':mlx_lm.__version__,'ram_gb':round(mem/1e9,1)}}))\""],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        info = json.loads(r.stdout.strip())
        total_ram += info["ram_gb"]
        print(f"✅ {host}: MLX {info['mlx']} · mlx-lm {info['mlx_lm']} · {info['ram_gb']} GB")
    else:
        all_ok = False
        print(f"❌ {host}: {r.stderr.strip()}")

print(f"\nPooled RAM: {total_ram} GB — {'READY' if all_ok else 'NOT READY'}")
```

### Generate Hostfile

```bash
mlx.distributed_config \
  --backend ring \
  --over ethernet \
  --hosts mac-mini-1.local,mac-mini-2.local,mac-mini-3.local \
  --output ~/.mlx-cluster.json
```

### Launch Distributed LLM

```bash
# Distributed chat
mlx.launch \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  -- \
  /usr/local/bin/python3 -m mlx_lm.chat \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit

# Distributed server (OpenAI API on port 8080)
mlx.launch \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  -- \
  /usr/local/bin/python3 -m mlx_lm.server \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit \
    --port 8080
```

### Cluster RAM × Model Guide

```
2× 16 GB =  32 GB → Llama-3.1-8B full · Qwen2.5-14B-4bit
2× 24 GB =  48 GB → Llama-3.1-70B-4bit (tight)
4× 16 GB =  64 GB → Llama-3.1-70B-4bit (comfortable)
4× 24 GB =  96 GB → DeepSeek-R1-4bit · Llama-3.1-70B full precision
4× 32 GB = 128 GB → Any current model at full precision
```

### Distributed Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `Connection refused` port 22 | Remote Login off | `sudo systemsetup -setremotelogin on` |
| `Permission denied (publickey)` | SSH key not deployed | `ssh-copy-id user@node` |
| `python3: command not found` | Path mismatch | Use full absolute path in hostfile |
| Rank hangs / never connects | Firewall blocking ports | Allow Python through macOS firewall |
| `init()` returns `size=1` | Used `python3` not `mlx.launch` | Use `mlx.launch` |
| Slow despite cluster | Wi-Fi bottleneck | Switch to wired gigabit Ethernet |
