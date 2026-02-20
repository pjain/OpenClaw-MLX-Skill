# OpenClaw MLX Skill

> Run Apple MLX models locally — image generation, audio/TTS, vision analysis, and LLM text inference — on one Mac or a whole LAN cluster.

---

## Table of Contents

- [What This Skill Does](#what-this-skill-does)
- [Requirements](#requirements)
- [Single Mac Setup](#single-mac-setup)
  - [1. Install Python](#1-install-python)
  - [2. Install mflux (image generation)](#2-install-mflux-image-generation)
  - [3. Install other MLX packages](#3-install-other-mlx-packages)
  - [4. Hugging Face authentication](#4-hugging-face-authentication)
  - [5. Install the skill](#5-install-the-skill)
  - [6. Verify your setup](#6-verify-your-setup)
- [What You Can Ask For](#what-you-can-ask-for)
  - [Image generation](#image-generation)
  - [Audio & text-to-speech](#audio--text-to-speech)
  - [Vision / image analysis](#vision--image-analysis)
  - [LLM / text inference](#llm--text-inference)
  - [Pipelines](#pipelines)
- [Model Reference](#model-reference)
  - [Image models (mflux)](#image-models-mflux)
  - [LLM models (mlx-lm)](#llm-models-mlx-lm)
  - [Vision models (mlx-vlm)](#vision-models-mlx-vlm)
  - [Audio / TTS models](#audio--tts-models)
- [RAM Guide](#ram-guide)
- [LAN Cluster Setup](#lan-cluster-setup)
  - [How distributed inference works](#how-distributed-inference-works)
  - [Hardware recommendations](#hardware-recommendations)
  - [Step 1 — Prepare every node](#step-1--prepare-every-node)
  - [Step 2 — Enable Remote Login](#step-2--enable-remote-login)
  - [Step 3 — Set up passwordless SSH](#step-3--set-up-passwordless-ssh)
  - [Step 4 — Generate the hostfile](#step-4--generate-the-hostfile)
  - [Step 5 — Verify cluster readiness](#step-5--verify-cluster-readiness)
  - [Step 6 — Run distributed inference](#step-6--run-distributed-inference)
  - [Cluster RAM & model guide](#cluster-ram--model-guide)
- [OpenAI-Compatible LLM Server](#openai-compatible-llm-server)
- [Network & Firewall](#network--firewall)
- [Troubleshooting](#troubleshooting)
- [Extending the Skill](#extending-the-skill)
- [File Layout](#file-layout)
- [Quick Reference Card](#quick-reference-card)

---

## What This Skill Does

This skill gives OpenClaw (Claude) the ability to run AI models **entirely on your own hardware** using Apple's [MLX framework](https://github.com/ml-explore/mlx). No cloud required, no data leaving your network.

When you ask Claude to generate an image, speak text, describe a photo, or chat with a local LLM, the skill:

1. **Discovers** your environment — Apple Silicon, RAM, installed packages, cached models
2. **Selects** the best model for your hardware automatically
3. **Installs** any missing packages silently in the background
4. **Runs** the model and delivers the output
5. **Opens** results automatically on macOS

In a multi-Mac setup, the skill also **pools RAM across your LAN** using MLX distributed inference, allowing a cluster of Macs to run models that no single machine could fit.

### Supported capabilities

| Capability | Tool | Highlights |
|-----------|------|-----------|
| Image generation | `mflux` v0.16+ | Z-Image, FLUX.2, Flux-dev, Krea, Kontext, Qwen Image |
| Audio / TTS | `mlx-audio` | Kokoro-82M, Bark |
| Vision / image analysis | `mlx-vlm` | Qwen2-VL, LLaVA |
| LLM / chat | `mlx-lm` | Llama, Mistral, Qwen, DeepSeek, OpenAI-compatible server |
| Distributed LLM | `mlx-lm` + `mlx.launch` | Ring backend over LAN Ethernet |

---

## Requirements

### Minimum (single Mac)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Chip | Apple M1 | Apple M2 Pro or later |
| macOS | 13.5 Ventura | 14+ Sonoma or 15+ Sequoia |
| RAM | 8 GB | 24 GB or more |
| Free disk | 15 GB | 100 GB+ (models are large — Flux is ~34 GB) |
| Python | 3.10 | 3.11 or 3.12 |

> **Intel Macs are not supported.** MLX is Apple Silicon only. If you're on Intel, ask Claude to use a cloud provider like Replicate or fal.ai instead.

### For a LAN cluster (optional)

- Two or more Apple Silicon Macs on the same network
- Gigabit Ethernet strongly recommended (Wi-Fi works but is noticeably slower)
- Remote Login enabled on every node
- Passwordless SSH from the controller to all other nodes
- Identical Python binary path on every machine

---

## Single Mac Setup

### 1. Install Python

Python 3.10+ is required. mflux requires Python ≥ 3.10.

**Via Homebrew** (recommended for consistency across a cluster):

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Note the path — you'll need it for cluster setup
which python3
# e.g. /opt/homebrew/bin/python3  or  /usr/local/bin/python3

python3 --version  # should print 3.11.x
```

**Also install uv** (the preferred mflux installer — significantly faster):

```bash
brew install uv
# or:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install mflux (image generation)

mflux v0.16+ is a major upgrade over earlier versions. It adds Z-Image, FLUX.2 Klein,
Krea-dev, Kontext, and Qwen Image — and the Python API changed substantially.

**Recommended: install via uv tool** (keeps mflux isolated, easy to upgrade):

```bash
uv tool install --upgrade mflux --prerelease=allow

# Optional: enable faster HF downloads
uv tool install --upgrade mflux --with hf_transfer --prerelease=allow

# Install ZSH tab completions (optional but nice)
mflux-completions
exec zsh
```

**Alternative: install via pip**:

```bash
pip install -U mflux
```

**Verify**:

```bash
mflux-generate --help          # should show all options
mflux-generate-z-image-turbo --help   # new in v0.6+
```

> **If you're upgrading from mflux < 0.6**: The Python API changed. `Flux1.from_alias()` is
> now `Flux1.from_name()`, and the `Config` object is removed — parameters are passed directly
> to `generate_image()`. The CLI commands also changed (`flux-schnell` → `schnell`).
> See [Troubleshooting](#troubleshooting) if you hit errors after upgrading.

**mflux cache location** (changed in v0.6+):
- Model index/metadata: `~/Library/Caches/mflux/`
- HF model weights: `~/.cache/huggingface/` (unchanged)
- Override with: `export MFLUX_CACHE_DIR=/path/to/cache`

### 3. Install other MLX packages

```bash
# Audio / TTS
pip install mlx-audio soundfile

# Vision / image analysis
pip install mlx-vlm

# LLM inference
pip install mlx-lm

# Utilities
pip install huggingface_hub pillow
```

Install everything at once:

```bash
pip install mlx-audio soundfile mlx-vlm mlx-lm huggingface_hub pillow
```

Verify:

```bash
python3 -c "import mlx_audio; print('mlx-audio OK')"
python3 -c "import mlx_vlm; print('mlx-vlm OK')"
python3 -c "import mlx_lm; print('mlx-lm OK')"
```

### 4. Hugging Face authentication

Some models require a free Hugging Face account. Most do not — see the table below.

| Model family | Auth required? |
|-------------|---------------|
| Z-Image Turbo | ❌ No |
| FLUX.2 Klein | ❌ No (Apache 2.0) |
| Flux.1-schnell | ❌ No |
| Flux.1-dev | ✅ Yes (accept license on HF) |
| FLUX.2-dev | ✅ Yes |
| Kokoro TTS | ❌ No |
| mlx-community LLMs | ❌ No |
| mlx-community VLMs | ❌ No |

**To set up auth** (needed for Flux-dev and FLUX.2-dev):

1. Create a free account at [huggingface.co](https://huggingface.co/join)
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click **New token** → **Read** access → copy the token
4. Run: `huggingface-cli login` and paste your token

Or set it as an environment variable (add to `~/.zshrc` for persistence):

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

**Accept the Flux.1-dev license** (one-time):
1. Go to [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Click **Agree and access repository**

### 5. Install the skill

```bash
mkdir -p /Users/clawd/clawd/skills/mlx
cp SKILL.md /Users/clawd/clawd/skills/mlx/SKILL.md
```

Restart OpenClaw or reload skills if it supports hot-reload.

### 6. Verify your setup

Run this health check to confirm everything is working:

```bash
python3 - << 'EOF'
import subprocess, sys, platform, json, os

info = {}
is_apple_silicon = platform.machine() == "arm64" and platform.system() == "Darwin"
info["apple_silicon"] = is_apple_silicon

try:
    v = platform.mac_ver()[0]
    parts = list(map(int, v.split(".")))
    info["macos_ok"] = parts[0] > 13 or (parts[0] == 13 and parts[1] >= 5)
    info["macos_version"] = v
except:
    info["macos_ok"] = False

try:
    mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
    info["ram_gb"] = round(int(mem) / 1e9, 1)
except:
    info["ram_gb"] = "unknown"

packages = {
    "mflux": "mflux",
    "mlx_audio": "mlx-audio",
    "mlx_vlm": "mlx-vlm",
    "mlx_lm": "mlx-lm",
    "huggingface_hub": "huggingface_hub",
}
info["packages"] = {}
for mod, pkg in packages.items():
    try:
        m = __import__(mod)
        info["packages"][pkg] = getattr(m, "__version__", "installed")
    except ImportError:
        info["packages"][pkg] = "MISSING"

try:
    subprocess.check_output(["uv", "--version"])
    info["uv"] = "available"
except:
    info["uv"] = "not found (pip fallback will be used)"

print(json.dumps(info, indent=2))
EOF
```

A healthy setup looks like:

```json
{
  "apple_silicon": true,
  "macos_ok": true,
  "macos_version": "15.3.1",
  "ram_gb": 24.0,
  "packages": {
    "mflux": "0.16.5",
    "mlx-audio": "0.2.0",
    "mlx-vlm": "0.1.12",
    "mlx-lm": "0.22.1",
    "huggingface_hub": "0.27.0"
  },
  "uv": "uv 0.5.1"
}
```

If any package shows `"MISSING"`, install it: `pip install <package-name>`.

---

## What You Can Ask For

### Image generation

The skill defaults to **Z-Image Turbo** — the best speed/quality balance as of 2025
(6B params, ~9 steps, great realism). It automatically falls back to flux-schnell on
lower-RAM machines.

```
"Generate an image of a red fox in autumn leaves"
"Make a 1024×1024 photo of a futuristic Tokyo skyline at night"
"Create concept art for a fantasy castle on a floating island"
"Generate a photorealistic portrait of an astronaut on Mars"
"Make a picture of a cat — but make it look like a real photograph, not AI art"
"Edit this photo to change the background to a beach" [attach photo]
```

### Audio & text-to-speech

```
"Say 'Welcome to my presentation' in a warm female voice"
"Read this paragraph aloud: [paste text]"
"Generate audio for this narration script"
"Convert this text to speech with a British male voice"
```

Output is saved as `.wav` to `~/Desktop/mlx-outputs/` and played back automatically.

### Vision / image analysis

```
"Describe what's in this image" [attach photo]
"What text can you read in this screenshot?" [attach image]
"Analyze this chart and explain what it shows" [attach chart]
"What objects are in the foreground of this photo?" [attach image]
```

Images are analyzed locally — nothing leaves your machine.

### LLM / text inference

```
"Run Llama locally and explain the theory of relativity"
"Use a local model to write a Python function that sorts a list"
"Chat with a local LLM about my project"
"Start a local LLM server I can connect to"
"Use DeepSeek to reason through this problem step by step: [problem]"
```

### Pipelines

```
"Generate an image of a mountain lake, then describe it"
"Make a picture of a robot chef, then tell me what ingredients it's using"
"Create an image then have a local LLM write a story about what's in it"
```

---

## Model Reference

### Image models (mflux)

| Model | CLI command | RAM (q8) | Steps | Best for |
|-------|------------|----------|-------|---------|
| **Z-Image Turbo** ⭐ | `mflux-generate-z-image-turbo` | ~6 GB | 9 | Best all-rounder 2025 |
| FLUX.2 Klein | `mflux-generate --model flux2-klein` | ~5 GB | 4 | Fastest, Apache 2.0 |
| Flux.1-schnell | `mflux-generate --model schnell` | ~8 GB | 2–4 | Speed, quick drafts |
| Flux.1-dev | `mflux-generate --model dev` | ~8 GB | 20–25 | Quality |
| Krea-dev | `mflux-generate --model krea-dev` | ~8 GB | 25 | Photorealism, avoids AI look |
| FLUX.2-dev | `mflux-generate --model flux2-dev` | ~20 GB | 25 | FLUX.2 quality, needs auth |
| Qwen Image | `mflux-generate-qwen` | ~14 GB | 20 | Best prompt understanding |
| Kontext | `mflux-generate-kontext` | ~8 GB | 25 | Image editing with reference |

> **mflux supports LoRA, image-to-image, fill/inpainting, ControlNet, and Dreambooth fine-tuning.** These advanced features work with the same CLI pattern — see the [mflux GitHub](https://github.com/filipstrand/mflux) for full documentation.

### LLM models (mlx-lm)

All from the `mlx-community` namespace on Hugging Face.

| Model | RAM | Strength |
|-------|-----|---------|
| `Llama-3.2-3B-Instruct-4bit` | 4 GB | Lightest, fast |
| `Phi-3.5-mini-instruct-4bit` | 4 GB | Small but capable |
| `Mistral-7B-Instruct-v0.3-4bit` | 8 GB | Great general purpose |
| `Llama-3.1-8B-Instruct-4bit` | 8 GB | Best 8B overall |
| `Qwen2.5-14B-Instruct-4bit` | 14 GB | Strong reasoning |
| `Qwen2.5-32B-Instruct-4bit` | 32 GB | Excellent coding + reasoning |
| `Llama-3.1-70B-Instruct-4bit` | 40 GB | Near frontier quality |
| `DeepSeek-R1-0528-4bit` | 40+ GB | Best open-source reasoning |

### Vision models (mlx-vlm)

| Model | RAM | Strength |
|-------|-----|---------|
| `mlx-community/llava-1.5-7b-4bit` | 8 GB | General image Q&A |
| `mlx-community/Qwen2-VL-7B-Instruct-4bit` | 16 GB | Excellent OCR + detail |
| `mlx-community/Qwen2-VL-72B-Instruct-4bit` | 64 GB | Near-GPT4V quality |

### Audio / TTS models

| Model | RAM | Notes |
|-------|-----|-------|
| `prince-canuma/Kokoro-82M` | ~1 GB | Default — fast, high quality |
| `hexgrad/Kokoro-82M` | ~1 GB | Alternative Kokoro |
| `suno/bark-small` | ~2 GB | More expressive, slower |

**Kokoro voices**: `af` · `af_heart` · `af_bella` · `af_sarah` · `am_adam` · `bf_emma` · `bm_george`

---

## RAM Guide

### Single Mac

| Your RAM | Image | LLM | Vision |
|----------|-------|-----|--------|
| 8 GB | Z-Image q8 or schnell q8 | Llama-3.2-3B or Phi-3.5-mini | llava-1.5-7b |
| 16 GB | Z-Image q8 or flux-dev q8 | Llama-3.1-8B or Mistral-7B | Qwen2-VL-7B |
| 24 GB | Z-Image q8 or flux-dev q8 | Qwen2.5-14B or Llama-3.1-8B | Qwen2-VL-7B |
| 32 GB | flux-dev q8 or Qwen Image q6 | Qwen2.5-32B | Qwen2-VL-7B (full) |
| 64 GB | Any at q8 | Llama-3.1-70B-4bit | Qwen2-VL-72B-4bit |
| 96 GB+ | Full precision | DeepSeek-R1 | Qwen2-VL-72B (full) |

> **First run takes longer** because models download from Hugging Face. Sizes: Flux/Z-Image ~34 GB,
> Llama-3.1-8B ~5 GB, Llama-3.1-70B ~40 GB. Subsequent runs load from cache instantly.

---

## LAN Cluster Setup

A LAN cluster pools RAM across multiple Macs to run models that won't fit on any single machine.
This is optional — the skill works perfectly standalone — but is a major upgrade if you have
spare Apple Silicon hardware.

> **What distributes:** LLM inference (`mlx-lm`) fully supports multi-node sharding. Image
> generation (`mflux`) and vision (`mlx-vlm`) don't yet support multi-node — for those, the
> skill uses the single node with the most RAM.

### How distributed inference works

MLX uses **tensor parallelism**: each node holds a shard of the model's weight matrices.
On every forward pass, nodes compute their shard and exchange intermediate results via
collective operations (`all_reduce`) over TCP/IP. You run `mlx.launch` on the controller
node — it SSHes to all others, starts the process, and forwards output back to your terminal.

```
                    Gigabit Ethernet
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │ Node 0  │◄─────►│ Node 1  │◄─────►│ Node 2  │
   │ Rank 0  │       │ Rank 1  │       │ Rank 2  │
   │(ctrl)   │       │         │       │         │
   │Layers   │       │Layers   │       │Layers   │
   │ 0–N/3   │       │N/3–2N/3 │       │2N/3–N   │
   └─────────┘       └─────────┘       └─────────┘
        │
   mlx.launch
   (run here)
```

### Hardware recommendations

| Cluster | Pooled RAM | Unlocks |
|---------|-----------|---------|
| 2× Mac Mini M4 16 GB | 32 GB | Llama-3.1-8B full, Qwen2.5-14B-4bit |
| 2× Mac Mini M4 24 GB | 48 GB | Llama-3.1-70B-4bit (tight) |
| 4× Mac Mini M4 16 GB | 64 GB | Llama-3.1-70B-4bit (comfortable) |
| 4× Mac Mini M4 24 GB | 96 GB | DeepSeek-R1-4bit, 70B full precision |
| 4× Mac Studio M2 Ultra 192 GB | 768 GB | Everything |

**Network**: Wired gigabit Ethernet is strongly preferred. The ring backend sends large
tensors on every forward pass. Wi-Fi works but introduces latency. 2.5 Gbps or 10 Gbps
links noticeably improve throughput.

---

### Step 1 — Prepare every node

Do this on **every Mac in the cluster**, including the controller.

**a) Install the same Python version at the same path on all nodes**

This is the most common failure point. Every node must have Python at the **exact same path**.

```bash
# Install Python via Homebrew (recommended for consistency)
brew install python@3.11

# Note the path
which python3
# Should be /opt/homebrew/bin/python3 on all M-series Macs
```

**b) Install MLX packages on every node**

```bash
# Core requirement for distributed LLM
pip install mlx mlx-lm huggingface_hub

# For image/vision if those nodes will serve those workloads
pip install mlx-audio mlx-vlm pillow soundfile
uv tool install --upgrade mflux --prerelease=allow
```

**c) Verify on each node**

```bash
python3 -c "import mlx; import mlx_lm; print('Node ready:', mlx.__version__)"
```

**d) Confirm hostname resolution**

```bash
# From controller, ping each node
ping -c 1 mac-mini-2.local
ping -c 1 mac-mini-3.local
```

If ping fails, use IP addresses instead (check your router's DHCP table, or run
`arp -a` from the controller).

---

### Step 2 — Enable Remote Login

On **every node** (including optionally the controller):

**Via System Settings:**
1. Open **System Settings**
2. Go to **General → Sharing**
3. Enable **Remote Login**
4. Set **Allow access** to all users or your specific account

**Via Terminal** (can run remotely or locally):

```bash
# Enable Remote Login
sudo systemsetup -setremotelogin on

# Verify
sudo systemsetup -getremotelogin
# Prints: Remote Login: On
```

---

### Step 3 — Set up passwordless SSH

`mlx.launch` SSHes into all nodes automatically and cannot prompt for passwords. Set up
key-based auth before your first distributed run.

Run all of this on the **controller node**:

```bash
# 1. Generate a cluster-specific SSH key
ssh-keygen -t ed25519 -C "mlx-cluster" -f ~/.ssh/id_mlx -N ""

# 2. Add to your SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_mlx

# 3. Copy the public key to each node (enter that node's password once)
ssh-copy-id -i ~/.ssh/id_mlx.pub your-username@mac-mini-2.local
ssh-copy-id -i ~/.ssh/id_mlx.pub your-username@mac-mini-3.local
ssh-copy-id -i ~/.ssh/id_mlx.pub your-username@mac-mini-4.local

# 4. Test — must NOT prompt for a password
ssh mac-mini-2.local "echo 'Node 2: SSH OK'"
ssh mac-mini-3.local "echo 'Node 3: SSH OK'"
```

To make the key persistent across reboots, add to `~/.ssh/config`:

```
Host *.local
    IdentityFile ~/.ssh/id_mlx
    StrictHostKeyChecking no
    ServerAliveInterval 30
```

---

### Step 4 — Generate the hostfile

`mlx.launch` needs a JSON hostfile describing all cluster nodes.

**Using the built-in MLX tool** (easiest):

```bash
mlx.distributed_config \
  --backend ring \
  --over ethernet \
  --hosts mac-mini-1.local,mac-mini-2.local,mac-mini-3.local \
  --output ~/.mlx-cluster.json
```

**Manual format** (use IPs if `.local` hostname resolution is unreliable):

```json
[
  { "ssh": "192.168.1.10", "ips": ["192.168.1.10"], "python": "/opt/homebrew/bin/python3" },
  { "ssh": "192.168.1.11", "ips": ["192.168.1.11"], "python": "/opt/homebrew/bin/python3" },
  { "ssh": "192.168.1.12", "ips": ["192.168.1.12"], "python": "/opt/homebrew/bin/python3" }
]
```

> The `python` path must be **identical across all nodes** and point to the Python with
> `mlx` and `mlx-lm` installed.

---

### Step 5 — Verify cluster readiness

Save this as `check_cluster.py` and run it before your first distributed inference:

```python
import subprocess, json

HOSTS = [
    "mac-mini-2.local",
    "mac-mini-3.local",
    "mac-mini-4.local",
]
PYTHON_PATH = "/opt/homebrew/bin/python3"  # must match all nodes

def check_node(host):
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host,
         f"{PYTHON_PATH} -c \""
         f"import mlx, mlx_lm, subprocess, json;"
         f"mem=int(subprocess.check_output(['sysctl','-n','hw.memsize']).decode());"
         f"ip=subprocess.getoutput('ipconfig getifaddr en0');"
         f"print(json.dumps({{'mlx':mlx.__version__,'mlx_lm':mlx_lm.__version__,"
         f"'ram_gb':round(mem/1e9,1),'ip':ip}}))"
         f"\""],
        capture_output=True, text=True, timeout=10
    )
    if r.returncode == 0:
        info = json.loads(r.stdout.strip())
        info["host"] = host
        info["ssh"] = True
        return info
    return {"host": host, "ssh": False, "error": r.stderr.strip()}

total_ram = 0
all_ok = True
header = f"{'Host':<26} {'SSH':>5} {'MLX':>8} {'mlx-lm':>8} {'RAM':>8} {'IP'}"
print(header)
print("─" * 75)
for host in HOSTS:
    info = check_node(host)
    ok = info.get("ssh") and info.get("mlx") and info.get("mlx_lm")
    ram = info.get("ram_gb", 0)
    total_ram += ram
    status = "✅" if ok else "❌"
    print(f"{status} {host:<24} {str(info.get('ssh')):>5} "
          f"{info.get('mlx','—'):>8} {info.get('mlx_lm','—'):>8} "
          f"{str(ram)+'GB':>8} {info.get('ip','?')}")
    if not ok:
        all_ok = False
        if info.get("error"):
            print(f"   └─ {info['error']}")

print("─" * 75)
print(f"Pooled RAM: {total_ram} GB")
print(f"Status: {'READY ✅' if all_ok else 'NOT READY ❌ — fix issues above'}")
```

```bash
python3 check_cluster.py
```

Expected output:

```
Host                       SSH      MLX   mlx-lm      RAM IP
───────────────────────────────────────────────────────────────────────────
✅ mac-mini-2.local         True   0.30.6   0.22.1    24.0GB 192.168.1.11
✅ mac-mini-3.local         True   0.30.6   0.22.1    24.0GB 192.168.1.12
✅ mac-mini-4.local         True   0.30.6   0.22.1    16.0GB 192.168.1.13
───────────────────────────────────────────────────────────────────────────
Pooled RAM: 64.0 GB
Status: READY ✅
```

---

### Step 6 — Run distributed inference

Once the cluster is verified, launch from the **controller node**:

```bash
# Distributed interactive chat
mlx.launch \
  --verbose \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  -- \
  /opt/homebrew/bin/python3 -m mlx_lm.chat \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit

# Distributed one-shot generation
mlx.launch \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  -- \
  /opt/homebrew/bin/python3 -m mlx_lm.generate \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit \
    --prompt "Explain tensor parallelism in simple terms" \
    --max-tokens 500

# Distributed OpenAI-compatible server
mlx.launch \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  -- \
  /opt/homebrew/bin/python3 -m mlx_lm.server \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit \
    --port 8080
```

**From OpenClaw**, just ask naturally:

```
"Use my cluster to run Llama 70B and explain quantum entanglement"
"Start a distributed LLM server across all my Macs"
```

The skill detects the hostfile, verifies nodes, and launches automatically.

---

### Cluster RAM & model guide

| Cluster | RAM | Models unlocked |
|---------|-----|----------------|
| 2× 16 GB | 32 GB | Llama-3.1-8B (full), Qwen2.5-14B-4bit |
| 2× 24 GB | 48 GB | Llama-3.1-70B-4bit (tight) |
| 2× 32 GB | 64 GB | Llama-3.1-70B-4bit (comfortable) |
| 4× 16 GB | 64 GB | Llama-3.1-70B-4bit, DeepSeek-R1-32B |
| 4× 24 GB | 96 GB | DeepSeek-R1-0528-4bit, 70B full precision |
| 4× 32 GB | 128 GB | Any current model at full precision |

**Best distributed LLMs** (tensor-parallel friendly):

- `mlx-community/Llama-3.1-70B-Instruct-4bit` — 40 GB, flagship general model
- `mlx-community/DeepSeek-R1-0528-4bit` — outstanding reasoning
- `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` — MoE distributes efficiently
- `mlx-community/Qwen2.5-72B-Instruct-4bit` — excellent multilingual + coding

---

## OpenAI-Compatible LLM Server

`mlx-lm` includes a built-in server with an OpenAI-compatible API. Any tool that speaks
the OpenAI API (LangChain, Open WebUI, curl, your own Python code) can use it.

```bash
# Start server on port 8080
mlx_lm.server \
  --model mlx-community/Llama-3.1-8B-Instruct-4bit \
  --port 8080

# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**With Python** (using the openai library):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.1-8B-Instruct-4bit",
    messages=[{"role": "user", "content": "Write a haiku about Apple Silicon."}],
)
print(response.choices[0].message.content)
```

Run the server distributed across your cluster to expose the 70B model on your LAN:

```bash
mlx.launch --backend ring --hostfile ~/.mlx-cluster.json -- \
  /opt/homebrew/bin/python3 -m mlx_lm.server \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit \
    --host 0.0.0.0 --port 8080
```

---

## Network & Firewall

macOS's firewall may block the ports MLX uses for inter-node communication. If nodes
hang waiting for each other during distributed inference:

```bash
# Allow Python through the application firewall (run on each node)
/usr/libexec/ApplicationFirewall/socketfilterfw --add /opt/homebrew/bin/python3
/usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /opt/homebrew/bin/python3

# Or temporarily disable for testing (re-enable after)
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
# ...test...
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
```

The ring backend uses ephemeral ports negotiated at startup. Whitelisting the Python
binary is simpler than trying to pin specific ports.

---

## Troubleshooting

### mflux API changes (upgrading from < v0.6)

| Old API (broken) | New API (correct) |
|----------------|------------------|
| `Flux1.from_alias(alias="flux-schnell")` | `Flux1.from_name(model_name="schnell")` |
| `from mflux import Flux1, Config` | `from mflux.models.flux.variants.txt2img.flux import Flux1` |
| `Config(num_inference_steps=4, ...)` | Pass `num_inference_steps=4` directly to `generate_image()` |
| `--model flux-schnell` (CLI) | `--model schnell` (CLI) |

If you see `AttributeError: type object 'Flux1' has no attribute 'from_alias'` or
`ImportError: cannot import name 'Config'`, upgrade mflux:

```bash
uv tool upgrade mflux --prerelease=allow
# or: pip install -U mflux
```

### Single Mac issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `machine != arm64` | Intel Mac | MLX not supported — use cloud APIs |
| `RuntimeError: Out of memory` | Model too large | Add `-q 4` or choose smaller model |
| Process killed (SIGKILL) | macOS OOM killer | Add `--low-ram` flag to mflux CLI |
| `403 Forbidden` downloading model | HF auth required | `huggingface-cli login` |
| `Repository not found` | HF auth or wrong model ID | Check model page, accept license |
| Very slow (>10 min) | No quantization | Add `-q 8` |
| Black / corrupted image | Stale mflux or MLX version | `pip install -U mflux mlx` |
| First run takes 20–60 min | Downloading 34 GB model | Normal — models are large |
| VLM output is garbled | Wrong prompt template | Use `apply_chat_template` from mlx_vlm |
| `mlx_lm.server` not found | Old mlx-lm | `pip install -U mlx-lm` |

### Cluster / distributed issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on port 22 | Remote Login not enabled | `sudo systemsetup -setremotelogin on` |
| `Permission denied (publickey)` | SSH key not deployed | `ssh-copy-id user@node` |
| `python3: command not found` | Path mismatch between nodes | Use full absolute path in hostfile |
| Ranks hang / never connect | Firewall blocking | Allow Python through macOS firewall |
| `init()` returns `size=1` | Ran with `python3`, not `mlx.launch` | Use `mlx.launch` launcher |
| Slow generation despite cluster | Wi-Fi bottleneck | Switch to wired gigabit Ethernet |
| `.local` hostname fails | mDNS / Bonjour issue | Use IP addresses in hostfile |
| Unequal load across nodes | Heterogeneous RAM | Normal — MLX handles this automatically |

### Pre-download models to avoid repeated cluster downloads

Instead of downloading the same 40 GB model on every node:

```bash
# 1. Download once on the controller
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Llama-3.1-70B-Instruct-4bit')
"

# 2. Sync to each node
rsync -avz --progress \
  ~/.cache/huggingface/hub/models--mlx-community--Llama-3.1-70B-Instruct-4bit/ \
  your-username@mac-mini-2.local:~/.cache/huggingface/hub/models--mlx-community--Llama-3.1-70B-Instruct-4bit/
```

Alternatively, share the HF cache over NFS so all nodes read from the same location.

---

## Extending the Skill

The skill is designed to be extended as new MLX packages and model families appear.
To add a new capability, edit `SKILL.md`:

1. **Add the package** to the `ensure(...)` call in Step 3
2. **Add to the category table** in Step 1 if it's a new capability
3. **Add RAM requirements** to the Step 2 RAM guides
4. **Add usage code** to Step 5 under a new subsection
5. **Add to the tracked projects table** at the bottom
6. **Add error patterns** to the Error Handling section

### Tracked MLX projects

| Project | PyPI | HF Namespace | Status | Notes |
|---------|------|--------------|--------|-------|
| mflux (images) | `mflux` | `black-forest-labs`, `Tongyi-MAI` | ✅ v0.16+ | Primary image gen |
| Audio / TTS | `mlx-audio` | `prince-canuma` | ✅ Stable | TTS, music |
| Vision / VLM | `mlx-vlm` | `mlx-community` | ✅ Stable | Image analysis |
| LLM inference | `mlx-lm` | `mlx-community` | ✅ Stable | Chat, server, distributed |
| MLX core | `mlx` | — | ✅ Stable | Dependency |
| Whisper STT | `mlx-whisper` | `mlx-community` | ✅ Stable | Speech-to-text |
| Stable Audio | via `mlx-audio` | `stabilityai` | 🧪 Experimental | Music generation |

---

## File Layout

```
/Users/clawd/clawd/skills/mlx/
├── SKILL.md              ← The skill (read by OpenClaw/Claude)
└── README.md             ← This file

~/Library/Caches/mflux/   ← mflux model index/metadata cache (v0.6+)
~/.cache/huggingface/     ← HF model weights (all packages)
~/Desktop/mlx-outputs/    ← Default output directory for generated files

~/.mlx-cluster.json       ← Cluster hostfile (created during cluster setup)
~/.ssh/id_mlx             ← SSH key for cluster auth
check_cluster.py          ← Cluster health check script (save anywhere)
```

Override cache locations:
```bash
export MFLUX_CACHE_DIR=/Volumes/SSD/mflux-cache
export HF_HOME=/Volumes/SSD/.cache/huggingface
```

---

## Quick Reference Card

```bash
# ── Image Generation ────────────────────────────────────────────────────────

# Best all-rounder (recommended default)
mflux-generate-z-image-turbo --prompt "your prompt" \
  --steps 9 --seed 42 -q 8 --width 1024 --height 1024

# Fastest (FLUX.2 Klein, Apache 2.0)
mflux-generate --model flux2-klein --prompt "your prompt" --steps 4 -q 8

# High quality (Flux.1-dev)
mflux-generate --model dev --prompt "your prompt" --steps 25 -q 8

# Photorealistic (avoids AI look)
mflux-generate --model krea-dev --prompt "A photo of a dog" --steps 25 -q 8

# Image editing with Kontext
mflux-generate-kontext --image-path photo.jpg --prompt "change background to beach" --steps 25

# Best prompt understanding
mflux-generate-qwen --prompt "your prompt" --steps 20 -q 6

# ── Audio / TTS ──────────────────────────────────────────────────────────────

python3 -c "
from mlx_audio.tts.generate import generate_audio; import soundfile as sf
audio, sr = generate_audio('Hello world', model='prince-canuma/Kokoro-82M', voice='af_heart')
sf.write('out.wav', audio, sr)
import subprocess; subprocess.Popen(['afplay', 'out.wav'])
"

# ── LLM Inference ────────────────────────────────────────────────────────────

# One-shot
mlx_lm.generate --model mlx-community/Llama-3.1-8B-Instruct-4bit --prompt "Hello"

# Interactive chat
mlx_lm.chat --model mlx-community/Llama-3.1-8B-Instruct-4bit

# OpenAI-compatible server
mlx_lm.server --model mlx-community/Llama-3.1-8B-Instruct-4bit --port 8080

# ── Vision ───────────────────────────────────────────────────────────────────

python3 -c "
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
m, p = load('mlx-community/Qwen2-VL-7B-Instruct-4bit')
c = load_config('mlx-community/Qwen2-VL-7B-Instruct-4bit')
f = apply_chat_template(p, c, 'Describe this image.', num_images=1)
print(generate(m, p, 'image.jpg', f, verbose=False))
"

# ── Cluster ──────────────────────────────────────────────────────────────────

# Health check
python3 check_cluster.py

# Generate hostfile
mlx.distributed_config --backend ring --over ethernet \
  --hosts node1.local,node2.local --output ~/.mlx-cluster.json

# Distributed 70B chat
mlx.launch --backend ring --hostfile ~/.mlx-cluster.json \
  --env MLX_METAL_FAST_SYNCH=1 -- \
  /opt/homebrew/bin/python3 -m mlx_lm.chat \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit

# Distributed OpenAI server
mlx.launch --backend ring --hostfile ~/.mlx-cluster.json -- \
  /opt/homebrew/bin/python3 -m mlx_lm.server \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit --host 0.0.0.0 --port 8080
```

---

*Skill version: 2.0 · mflux v0.16+ · mlx-lm v0.22+ · Requires macOS 13.5+ on Apple Silicon*
