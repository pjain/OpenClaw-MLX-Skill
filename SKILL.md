---
name: mlx
description: >
  Run local Apple MLX-based models for image generation (Flux, SDXL), audio/TTS (mlx-audio),
  and vision/multimodal models (mlx-vlm). Handles environment discovery, silent dependency
  installation, model downloads, and execution. Use whenever the user asks to generate images,
  audio, or describe/analyze images locally using MLX.
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

packages_to_check = ["mlx", "mflux", "mlx_audio", "mlx_vlm",
                     "huggingface_hub", "numpy", "pillow", "soundfile"]
installed = {}
for pkg in packages_to_check:
    try:
        mod = __import__(pkg)
        installed[pkg] = getattr(mod, "__version__", "installed")
    except ImportError:
        installed[pkg] = None
info["installed_packages"] = installed

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

print(json.dumps(info, indent=2))
```

Parse this for:
- `apple_silicon` + `macos_ok` → MLX is viable; if false, stop and inform the user
- `ram_gb` → determines safe model sizes (see Step 2)
- `installed_packages` → what the silent-installer needs to handle
- `downloaded_models` → models already cached (skip re-download)
- `hf_user` → whether HF auth is already set up

**If not Apple Silicon**: Inform the user MLX requires Apple Silicon Macs. Suggest cloud
alternatives like Replicate or fal.ai for image/audio generation.

---

## Step 1 — Map Request to Category

| Category | User says... | Package | Notes |
|----------|-------------|---------|-------|
| **Image generation** | "generate an image", "make art of X", "create a picture" | `mflux` | Primary focus |
| **Audio / TTS** | "say this", "speak", "generate audio", "text to speech" | `mlx-audio` | Kokoro default |
| **Vision / VLM** | "describe this image", "what's in this photo", "analyze this" | `mlx-vlm` | Needs image input |
| **Music generation** | "generate music", "make a song" | `mlx-audio` (stable-audio) | Experimental |
| **Image → Vision pipeline** | "generate then describe", "make and analyze" | `mflux` + `mlx-vlm` | Chain both |

---

## Step 2 — RAM-Based Model Selection

Use `ram_gb` from discovery to select appropriately. Never load a model that will OOM.

```
8 GB   → Flux.1-schnell q4/q8 · Kokoro TTS · llava-1.5-7b-4bit
16 GB  → Flux.1-dev q8 · Kokoro TTS · llava-1.5-13b or Qwen2-VL-7B-4bit
32 GB  → Flux.1-dev q8 (comfortable) · Qwen2-VL-7B-4bit
64 GB  → Flux.1-dev full precision · Qwen2-VL-72B-4bit
96 GB+ → Any model at full precision
```

Always tell the user which model was selected and why (RAM-based).

---

## Step 3 — Silent Auto-Install

**Always install missing packages silently without asking the user.** Include this helper at
the top of every generated script:

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

Usage before imports:
```python
ensure("mflux", "pillow")             # image generation
ensure("mlx-audio", "soundfile")      # audio / TTS
ensure("mlx-vlm")                     # vision
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

**Requires HF login**: `black-forest-labs/FLUX.1-dev`
**No login needed**: Flux.1-schnell via mflux, Kokoro TTS, most `mlx-community` VLMs

---

## Step 5 — Execution

### Image Generation (mflux)

```python
ensure("mflux", "pillow")

import os
from mflux import Flux1, Config

out_dir = os.path.expanduser("~/Desktop/mlx-outputs")
os.makedirs(out_dir, exist_ok=True)

# Set these from Step 0 discovery
ram_gb = 16  # replace with actual value
model_name = "flux-schnell" if ram_gb < 16 else "flux-dev"
steps     = 4 if model_name == "flux-schnell" else 20
quantize  = 4 if ram_gb <= 8 else 8

flux = Flux1.from_alias(alias=model_name, quantize=quantize)
image = flux.generate_image(
    seed=42,
    prompt="A photorealistic cat sitting on a misty mountain",
    config=Config(num_inference_steps=steps, height=1024, width=1024),
)
output_path = os.path.join(out_dir, "output.png")
image.save(path=output_path, export_json_metadata=False)
print(f"Saved: {output_path}")

import subprocess
subprocess.Popen(["open", output_path])  # preview on macOS
```

**Quantization guide**:
- `quantize=4` → ~4 GB, fastest, slight quality loss
- `quantize=8` → ~8 GB, balanced (recommended default)
- no quantize → full precision, most RAM, best quality

**CLI shortcut**:
```bash
mflux-generate --model flux-schnell --prompt "your prompt" \
  --steps 4 --width 1024 --height 1024 --quantize 8 \
  --output ~/Desktop/mlx-outputs/output.png
```

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
subprocess.Popen(["afplay", output_path])  # play on macOS
```

**TTS Models** (lightest → best quality):
| Model | RAM | Notes |
|-------|-----|-------|
| `prince-canuma/Kokoro-82M` | ~1 GB | Default — fast, good quality |
| `hexgrad/Kokoro-82M` | ~1 GB | Alternative Kokoro variant |
| `suno/bark-small` | ~2 GB | More expressive, slower |

**Kokoro voices**: `af` · `af_heart` · `af_bella` · `af_sarah` · `am_adam` · `bf_emma` · `bm_george`

---

### Vision / VLM (mlx-vlm)

```python
ensure("mlx-vlm", "pillow")

import os
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# RAM-based model selection — replace ram_gb with value from Step 0
ram_gb = 16
if ram_gb >= 64:
    model_id = "mlx-community/Qwen2-VL-72B-Instruct-4bit"
elif ram_gb >= 16:
    model_id = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
else:
    model_id = "mlx-community/llava-1.5-7b-4bit"

model, processor = load(model_id)
config = load_config(model_id)

image_path = "path/to/image.jpg"  # local path or URL
prompt = "Describe this image in detail."

formatted = apply_chat_template(processor, config, prompt, num_images=1)
response = generate(model, processor, image_path, formatted, verbose=False)
print(response)
```

**VLM Recommendations**:
| Model | RAM | Strength |
|-------|-----|---------|
| `mlx-community/llava-1.5-7b-4bit` | 8 GB | General vision Q&A |
| `mlx-community/llava-1.5-13b-4bit` | 16 GB | Better reasoning |
| `mlx-community/Qwen2-VL-7B-Instruct-4bit` | 16 GB | Excellent OCR + detail |
| `mlx-community/Qwen2-VL-72B-Instruct-4bit` | 64 GB | Near-GPT4V quality |

---

### Combined: Image → Vision Pipeline

```python
ensure("mflux", "mlx-vlm", "pillow")

import os
from mflux import Flux1, Config
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

out_dir = os.path.expanduser("~/Desktop/mlx-outputs")
os.makedirs(out_dir, exist_ok=True)
img_path = os.path.join(out_dir, "generated.png")

# 1. Generate
flux = Flux1.from_alias(alias="flux-schnell", quantize=8)
image = flux.generate_image(
    seed=42,
    prompt="A serene Japanese garden at dawn",
    config=Config(num_inference_steps=4, height=1024, width=1024),
)
image.save(path=img_path, export_json_metadata=False)
print(f"Image saved: {img_path}")

# 2. Analyze
model, processor = load("mlx-community/llava-1.5-7b-4bit")
cfg = load_config("mlx-community/llava-1.5-7b-4bit")
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
| `403 Forbidden` / `Repository not found` | HF auth required | `huggingface-cli login` |
| `ModuleNotFoundError` | Package missing | `ensure(...)` auto-handles on next run |
| Process killed (SIGKILL) | macOS OOM killer | Switch to quantize=4 or smaller model |
| Very slow (>5 min) | No quantization | Add quantize=8 |
| Black / corrupted image | Stale mflux version | `pip install -U mflux` |
| VLM garbled output | Wrong prompt template | Use `apply_chat_template` from mlx_vlm |

---

## Extensibility — Adding New MLX Models

When a new MLX project appears, add it here:

1. **Find the PyPI package** — check the project's GitHub README
2. **Add `ensure("new-package")`** to Step 5 in the relevant section
3. **Add to the category table** in Step 1 if it's a new capability
4. **Add RAM requirements** to Step 2 if known
5. **Add to the tracked projects table** below
6. **Add error patterns** to the Error Handling table

### Tracked MLX Projects

| Project | PyPI | HF Namespace | Status | Priority |
|---------|------|--------------|--------|----------|
| Flux image generation | `mflux` | `black-forest-labs` | ✅ Stable | ⭐ Primary |
| Audio / TTS | `mlx-audio` | `prince-canuma` | ✅ Stable | ⭐ Primary |
| Vision / VLM | `mlx-vlm` | `mlx-community` | ✅ Stable | ⭐ Primary |
| MLX core | `mlx` | — | ✅ Stable | Dependency |
| Whisper STT | `mlx-whisper` | `mlx-community` | ✅ Stable | Extend when needed |
| Stable Audio | via `mlx-audio` | `stabilityai` | 🧪 Experimental | Music generation |
| SDXL | `mlx-stable-diffusion` | `apple` | 🧪 Experimental | Alt image gen |

---

## Quick Reference

```bash
# Image — Flux CLI
mflux-generate --model flux-schnell --prompt "your prompt" \
  --steps 4 --quantize 8 --output ~/Desktop/mlx-outputs/out.png

# Audio — TTS one-liner
python3 -c "
from mlx_audio.tts.generate import generate_audio
import soundfile as sf
audio, sr = generate_audio('Hello world', model='prince-canuma/Kokoro-82M', voice='af_heart')
sf.write('out.wav', audio, sr)
import subprocess; subprocess.Popen(['afplay', 'out.wav'])
"

# Vision — describe an image
python3 -c "
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
m, p = load('mlx-community/llava-1.5-7b-4bit')
c = load_config('mlx-community/llava-1.5-7b-4bit')
f = apply_chat_template(p, c, 'What is in this image?', num_images=1)
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
    │       RAM < 8 GB    →  flux-schnell, quantize=4
    │       RAM 8-15 GB   →  flux-schnell, quantize=8
    │       RAM 16+ GB    →  flux-dev,     quantize=8
    │
    ├─ Audio / TTS
    │       Any RAM       →  Kokoro-82M (works everywhere)
    │       Want expressive → bark-small if RAM > 4 GB
    │
    ├─ Vision / VLM
    │       RAM 8-15 GB   →  llava-1.5-7b-4bit
    │       RAM 16-31 GB  →  Qwen2-VL-7B-Instruct-4bit
    │       RAM 64+ GB    →  Qwen2-VL-72B-Instruct-4bit
    │
    ├─ Image + Vision pipeline  →  mflux → mlx-vlm (see Step 5)
    │
    └─ New/unknown MLX model    →  Check mlx-community on HF, follow Extensibility guide
```

---

## Distributed Inference — LAN Cluster

MLX supports distributed inference across multiple Macs on a LAN using `mlx.launch`.
Model layers are sharded (tensor parallelism) across all nodes, letting you pool RAM across
machines to run models far larger than any single Mac could handle.

> **Combined RAM example**: 4× Mac Mini M4 with 24 GB each = 96 GB effective RAM,
> enough to run a 70B model at 4-bit quantization comfortably.

### How It Works

MLX distributes using **tensor parallelism** — each node holds a shard of the model's
weight matrices. On every forward pass, all nodes communicate intermediate results via
collective operations (`all_reduce`, `all_sum`). The ring backend does this over TCP/IP,
making it work on any LAN including Wi-Fi (though gigabit Ethernet or better is strongly
preferred for throughput).

### Backend Selection

| Backend | Transport | Requirements | Latency | Best For |
|---------|-----------|-------------|---------|----------|
| `ring` | Ethernet/Wi-Fi TCP | SSH + same Python path | Medium | **LAN clusters — use this** |
| `jaccl` | Thunderbolt 5 RDMA | macOS 26.2+, TB5 cables, recovery mode setup | Ultra-low | Directly connected Macs |
| `mpi` | TCP via OpenMPI | OpenMPI installed | Medium | Legacy setups |

For a general LAN (the common case), **always use the ring backend**.

---

### Prerequisites Checklist

Run this on the **controller node** to verify readiness before attempting distributed launch:

```python
import subprocess, json, os, sys

def check_node(host, python_path=None):
    """SSH to a node and verify it's ready for MLX distributed inference."""
    results = {}

    # SSH reachable?
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes",
                        host, "echo ok"], capture_output=True, text=True)
    results["ssh_ok"] = r.returncode == 0

    if not results["ssh_ok"]:
        results["error"] = r.stderr.strip()
        return results

    # Apple Silicon?
    r = subprocess.run(["ssh", host, "uname -m"], capture_output=True, text=True)
    results["apple_silicon"] = r.stdout.strip() == "arm64"

    # Python path
    py = python_path or "python3"
    r = subprocess.run(["ssh", host, f"{py} -c 'import mlx; print(mlx.__version__)'"],
                       capture_output=True, text=True)
    results["mlx_installed"] = r.returncode == 0
    results["mlx_version"] = r.stdout.strip() if r.returncode == 0 else None

    # mlx-lm installed?
    r = subprocess.run(["ssh", host, f"{py} -c 'import mlx_lm; print(mlx_lm.__version__)'"],
                       capture_output=True, text=True)
    results["mlx_lm_installed"] = r.returncode == 0

    # RAM
    r = subprocess.run(["ssh", host, "sysctl -n hw.memsize"], capture_output=True, text=True)
    if r.returncode == 0:
        results["ram_gb"] = round(int(r.stdout.strip()) / 1e9, 1)

    # IP on LAN
    r = subprocess.run(["ssh", host, "ipconfig getifaddr en0"], capture_output=True, text=True)
    results["lan_ip"] = r.stdout.strip() or None

    return results

# --- Configure these ---
HOSTS = ["mac-mini-1.local", "mac-mini-2.local", "mac-mini-3.local"]
PYTHON_PATH = "/usr/local/bin/python3"   # must be identical on all nodes
# ----------------------

print("Checking cluster readiness...\n")
total_ram = 0
all_ready = True
for host in HOSTS:
    r = check_node(host, PYTHON_PATH)
    ok = r.get("ssh_ok") and r.get("mlx_installed") and r.get("mlx_lm_installed")
    ram = r.get("ram_gb", 0)
    total_ram += ram
    status = "✅" if ok else "❌"
    print(f"{status} {host}: SSH={r.get('ssh_ok')} MLX={r.get('mlx_installed')} "
          f"mlx-lm={r.get('mlx_lm_installed')} RAM={ram}GB IP={r.get('lan_ip')}")
    if not ok:
        all_ready = False
        if r.get("error"):
            print(f"   Error: {r['error']}")

print(f"\nTotal pooled RAM: {total_ram} GB")
print(f"Cluster ready: {'YES' % all_ready if all_ready else 'NO — fix issues above first'}")
```

Fix any failures before proceeding:
- **SSH not ok**: Enable Remote Login on that Mac (`System Settings → General → Sharing → Remote Login`), then set up passwordless SSH keys
- **MLX not installed**: SSH in and `pip install mlx mlx-lm`
- **Python path mismatch**: All nodes must use the exact same Python binary path

---

### Step D1 — Set Up Passwordless SSH

Each node needs to be able to SSH to every other node without a password prompt.
Run this once from the controller, replacing hostnames:

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "mlx-cluster" -f ~/.ssh/id_mlx -N ""

# Copy key to each node
for HOST in mac-mini-1.local mac-mini-2.local mac-mini-3.local; do
    ssh-copy-id -i ~/.ssh/id_mlx.pub "$HOST"
done

# Verify (should not prompt for password)
for HOST in mac-mini-1.local mac-mini-2.local mac-mini-3.local; do
    ssh -i ~/.ssh/id_mlx "$HOST" "echo '$HOST: SSH OK'"
done
```

---

### Step D2 — Discover LAN Nodes Automatically

Don't hardcode IPs — scan the LAN for Macs with MLX installed:

```python
import subprocess, ipaddress, concurrent.futures, socket

def get_local_subnet():
    """Get the local /24 subnet to scan."""
    result = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True)
    ip = result.stdout.strip()
    if not ip:
        result = subprocess.run(["ipconfig", "getifaddr", "en1"], capture_output=True, text=True)
        ip = result.stdout.strip()
    if not ip:
        raise RuntimeError("Could not determine local IP address")
    # Return the /24 network
    parts = ip.rsplit(".", 1)
    return f"{parts[0]}.0/24", ip

def probe_host(ip, python_path="python3", timeout=2):
    """Check if a host has MLX installed and return its info."""
    try:
        # Try SSH
        r = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no", str(ip),
             f"{python_path} -c \"import mlx, mlx_lm, platform, json; "
             f"mem=int(__import__('subprocess').check_output(['sysctl','-n','hw.memsize']).decode()); "
             f"print(json.dumps({{'ip': str('{ip}'), 'ram_gb': round(mem/1e9,1), "
             f"'mlx': mlx.__version__, 'hostname': platform.node()}}))\""],
            capture_output=True, text=True, timeout=timeout + 1
        )
        if r.returncode == 0:
            import json
            return json.loads(r.stdout.strip())
    except:
        pass
    return None

def discover_mlx_nodes(python_path="python3", max_workers=50):
    """Scan the local subnet for Macs with MLX installed."""
    subnet, local_ip = get_local_subnet()
    print(f"Scanning {subnet} for MLX nodes (local IP: {local_ip})...")
    network = ipaddress.IPv4Network(subnet, strict=False)
    hosts = [str(ip) for ip in network.hosts()]

    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(probe_host, ip, python_path): ip for ip in hosts}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                found.append(result)
                print(f"  Found: {result['hostname']} ({result['ip']}) — "
                      f"{result['ram_gb']} GB RAM, MLX {result['mlx']}")

    found.sort(key=lambda x: x["ram_gb"], reverse=True)
    total_ram = sum(n["ram_gb"] for n in found)
    print(f"\nFound {len(found)} MLX nodes, {total_ram} GB pooled RAM")
    return found

# Run discovery
PYTHON_PATH = "/usr/local/bin/python3"  # adjust to match your nodes
nodes = discover_mlx_nodes(python_path=PYTHON_PATH)
```

---

### Step D3 — Generate Hostfile

`mlx.launch` needs a JSON hostfile describing each node. Generate it from discovered nodes:

```python
import json, subprocess

def generate_hostfile(nodes, python_path, output_path="~/.mlx-cluster.json"):
    """Generate an mlx.launch-compatible hostfile from discovered nodes."""
    import os
    output_path = os.path.expanduser(output_path)

    hostfile = []
    for i, node in enumerate(nodes):
        hostfile.append({
            "ssh": node["ip"],
            "ips": [node["ip"]],
            "python": python_path,
        })

    with open(output_path, "w") as f:
        json.dump(hostfile, f, indent=2)

    print(f"Hostfile written to: {output_path}")
    print(f"Contents:\n{json.dumps(hostfile, indent=2)}")
    return output_path

# OR: use MLX's built-in tool for Ethernet setup
# mlx.distributed_config --backend ring --over ethernet --hosts host1,host2,host3

# Generate from discovered nodes
PYTHON_PATH = "/usr/local/bin/python3"
hostfile_path = generate_hostfile(nodes, python_path=PYTHON_PATH)
```

**Or use the built-in MLX config tool** (simpler if all nodes use hostnames):
```bash
# Auto-generate hostfile for ring over Ethernet
mlx.distributed_config \
  --backend ring \
  --over ethernet \
  --hosts mac-mini-1.local,mac-mini-2.local,mac-mini-3.local \
  --output ~/.mlx-cluster.json
```

---

### Step D4 — Launch Distributed Inference

#### LLM Inference (mlx-lm, the primary distributed use case)

```bash
# Basic distributed chat
mlx.launch \
  --verbose \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  -- \
  /usr/local/bin/python3 -m mlx_lm.generate \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit \
    --prompt "Explain the theory of relativity" \
    --max-tokens 500

# Interactive distributed chat REPL
mlx.launch \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  -- \
  /usr/local/bin/python3 -m mlx_lm chat \
    --model mlx-community/DeepSeek-R1-0528-4bit
```

**With the performance env var** (recommended for LAN):
```bash
mlx.launch \
  --verbose \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  -- \
  /usr/local/bin/python3 -m mlx_lm chat \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit
```

#### Python API for distributed inference

```python
# This script must be launched via mlx.launch, not run directly
import mlx.core as mx
from mlx_lm import load, generate

# Initialize distributed group — automatically picks up rank/hostfile from env
world = mx.distributed.init()
rank = world.rank()
size = world.size()

if rank == 0:
    print(f"Running distributed inference across {size} nodes")

# Load model — mlx-lm handles sharding automatically when distributed
model, tokenizer = load("mlx-community/Llama-3.1-70B-Instruct-4bit")

# Only rank 0 needs to handle output
if rank == 0:
    response = generate(model, tokenizer, prompt="Hello!", max_tokens=200, verbose=True)
    print(response)
else:
    generate(model, tokenizer, prompt="Hello!", max_tokens=200, verbose=False)
```

Save as `distributed_infer.py` (same path on all nodes), then launch:
```bash
mlx.launch --backend ring --hostfile ~/.mlx-cluster.json \
  -- /usr/local/bin/python3 /path/to/distributed_infer.py
```

---

### Step D5 — Model Size vs Node Count Guide

Use pooled RAM to determine which models become accessible:

```
2 nodes × 16 GB =  32 GB → Llama-3.1-8B (full), Qwen2.5-14B-4bit
2 nodes × 24 GB =  48 GB → Llama-3.1-70B-4bit (tight), Mixtral-8x7B
2 nodes × 32 GB =  64 GB → Llama-3.1-70B-4bit (comfortable)
4 nodes × 16 GB =  64 GB → Llama-3.1-70B-4bit, DeepSeek-V3-4bit
4 nodes × 24 GB =  96 GB → DeepSeek-R1-4bit, Llama-3.1-70B full precision
4 nodes × 32 GB = 128 GB → Any current open-source model at full precision
8 nodes × 16 GB = 128 GB → Same as above
```

**Recommended distributed models** (tensor-parallel friendly):
- `mlx-community/Llama-3.1-70B-Instruct-4bit` — 40 GB, needs 2× 24 GB or 3× 16 GB
- `mlx-community/DeepSeek-R1-0528-4bit` — excellent reasoning, very large
- `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` — MoE, distributes efficiently
- `mlx-community/Qwen2.5-72B-Instruct-4bit` — great multilingual performance

---

### Distributed Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `ssh: connect to host X port 22: Connection refused` | Remote Login not enabled | System Settings → Sharing → Remote Login → On |
| `Permission denied (publickey)` | SSH key not on remote | Run `ssh-copy-id user@host` |
| `python3: command not found` | Python path differs between nodes | Use full path e.g. `/usr/local/bin/python3` |
| `KeyError: 'domain_uuid_key'` | Thunderbolt not properly connected (ring/TB mismatch) | Use `--backend ring --over ethernet` instead |
| Rank hangs / never connects | Firewall blocking ports | Allow ports 5000–5100 on all nodes |
| Slow generation despite multiple nodes | Wi-Fi bottleneck | Switch to wired gigabit Ethernet |
| `mx.distributed.init()` returns size=1 | Hostfile not found or env vars missing | Verify hostfile path, use `mlx.launch` not direct `python3` |
| Nodes get different model shard sizes | Unequal RAM across nodes | This is fine — MLX handles heterogeneous sharding |

---

### Distributed Architecture Summary

```
                    LAN (Ethernet/Wi-Fi)
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
    │ Node 0  │◄─────►│ Node 1  │◄─────►│ Node 2  │
    │ Rank 0  │       │ Rank 1  │       │ Rank 2  │
    │ (ctrl)  │       │         │       │         │
    │ Layers  │       │ Layers  │       │ Layers  │
    │  0-N/3  │       │ N/3-2N/3│       │ 2N/3-N  │
    └─────────┘       └─────────┘       └─────────┘
         │
    mlx.launch
    SSHes to all,
    coordinates,
    forwards output
    to terminal
```

All nodes participate equally in computation. Rank 0 is the coordinator — it's the one
you run `mlx.launch` from, and it collects and prints output.
