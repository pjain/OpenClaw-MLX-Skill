# OpenClaw MLX Skill

> Run Apple MLX models locally — image generation, audio/TTS, and vision analysis — on one Mac or a whole LAN cluster.

---

## Table of Contents

- [What This Skill Does](#what-this-skill-does)
- [Requirements](#requirements)
- [Single Mac Setup](#single-mac-setup)
  - [1. Install Python & pip](#1-install-python--pip)
  - [2. Install MLX packages](#2-install-mlx-packages)
  - [3. Hugging Face authentication](#3-hugging-face-authentication)
  - [4. Install the skill](#4-install-the-skill)
  - [5. Verify everything works](#5-verify-everything-works)
- [What You Can Ask For](#what-you-can-ask-for)
  - [Image generation](#image-generation)
  - [Audio & text-to-speech](#audio--text-to-speech)
  - [Vision / image analysis](#vision--image-analysis)
  - [Pipelines](#pipelines)
- [Model Selection & RAM Guide](#model-selection--ram-guide)
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
- [Network & Firewall](#network--firewall)
- [Troubleshooting](#troubleshooting)
- [Extending the Skill](#extending-the-skill)
- [File Layout](#file-layout)

---

## What This Skill Does

This skill gives OpenClaw (Claude) the ability to run AI models **entirely on your own hardware** using Apple's [MLX framework](https://github.com/ml-explore/mlx). No cloud, no API keys for generation, no data leaving your network.

When you ask Claude to generate an image, speak some text, or describe a photo, the skill:

1. **Discovers** your environment — checks for Apple Silicon, available RAM, installed packages, and cached models
2. **Selects** the best model for your hardware automatically
3. **Installs** any missing packages silently in the background
4. **Runs** the model and delivers the output file
5. **Opens** the result automatically on macOS

In a multi-Mac setup, it can also **pool RAM across your entire LAN** using MLX's distributed inference, letting a cluster of modest Macs run models that no single machine could fit.

---

## Requirements

### Minimum (single Mac)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Chip | Apple M1 | Apple M2 or later |
| macOS | 13.5 Ventura | 14+ Sonoma or 15+ Sequoia |
| RAM | 8 GB | 16 GB or more |
| Free disk | 10 GB | 50 GB+ (models are large) |
| Python | 3.9 | 3.11 or 3.12 |

> **Intel Macs are not supported.** MLX is Apple Silicon only. If you're on Intel, ask Claude to use a cloud provider like Replicate or fal.ai instead.

### For a LAN cluster (optional)

- Two or more Apple Silicon Macs on the same network
- Gigabit Ethernet strongly recommended (Wi-Fi works but is slower)
- Remote Login enabled on every node
- Passwordless SSH from the controller to all other nodes
- Identical Python path on every machine

---

## Single Mac Setup

### 1. Install Python & pip

The easiest way is via [Homebrew](https://brew.sh):

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Verify
python3 --version   # should print 3.11.x
which python3       # note this path — you'll need it for cluster setup
```

Alternatively, download the official installer from [python.org](https://www.python.org/downloads/macos/).

### 2. Install MLX packages

The skill auto-installs packages on demand, but pre-installing them means the first
request is faster and you can verify everything works before asking Claude to run anything.

```bash
# Core MLX framework
pip install mlx

# Image generation
pip install mflux pillow

# Audio / TTS
pip install mlx-audio soundfile

# Vision / multimodal
pip install mlx-vlm

# Utilities (always useful)
pip install huggingface_hub
```

To install everything at once:

```bash
pip install mlx mflux pillow mlx-audio soundfile mlx-vlm huggingface_hub
```

Verify the installs:

```bash
python3 -c "import mlx; print('MLX', mlx.__version__)"
python3 -c "import mflux; print('mflux OK')"
python3 -c "import mlx_audio; print('mlx-audio OK')"
python3 -c "import mlx_vlm; print('mlx-vlm OK')"
```

### 3. Hugging Face authentication

Some models require a free Hugging Face account to download. Flux.1-schnell (the fast
image model) and all TTS models work without an account, but Flux.1-dev (the high-quality
image model) requires one.

**Create a free account and token:**

1. Sign up at [huggingface.co](https://huggingface.co/join) if you don't have an account
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click **New token** → choose **Read** access → copy the token

**Log in from Terminal:**

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

Or set it as an environment variable (useful for automated setups):

```bash
# Add to ~/.zshrc or ~/.bash_profile for persistence
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

**Accept the Flux.1-dev license** (one-time, only needed for the high-quality model):

1. Go to [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Click **Agree and access repository**

You only need to do this once per account.

### 4. Install the skill

Copy `SKILL.md` into your OpenClaw skills directory:

```bash
mkdir -p /Users/clawd/clawd/skills/mlx
cp SKILL.md /Users/clawd/clawd/skills/mlx/SKILL.md
```

Then restart OpenClaw (or reload skills if it supports hot-reload).

### 5. Verify everything works

Run the built-in environment check to confirm your setup:

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

packages = ["mlx", "mflux", "mlx_audio", "mlx_vlm", "huggingface_hub"]
info["packages"] = {}
for pkg in packages:
    try:
        mod = __import__(pkg)
        info["packages"][pkg] = getattr(mod, "__version__", "installed")
    except ImportError:
        info["packages"][pkg] = "MISSING"

print(json.dumps(info, indent=2))
EOF
```

A healthy single-Mac setup looks like:

```json
{
  "apple_silicon": true,
  "macos_ok": true,
  "macos_version": "15.3.1",
  "ram_gb": 24.0,
  "packages": {
    "mlx": "0.30.6",
    "mflux": "0.4.1",
    "mlx_audio": "0.2.0",
    "mlx_vlm": "0.1.12",
    "huggingface_hub": "0.27.0"
  }
}
```

If any package shows `"MISSING"`, run `pip install <package-name>` (replace `_` with `-`).

---

## What You Can Ask For

Once the skill is installed, just talk to Claude naturally. Here are examples of things
that trigger the skill:

### Image generation

```
"Generate an image of a red fox sitting in autumn leaves"
"Make me a 1024×1024 picture of a futuristic Tokyo at night"
"Create concept art for a fantasy castle on a floating island"
"Generate a product photo of a wooden desk lamp on a white background"
```

Claude will automatically pick Flux.1-schnell for speed (4 steps, ~30 sec) or
Flux.1-dev for quality (20 steps, ~2–5 min) based on your RAM and the request.

Output is saved to `~/Desktop/mlx-outputs/` and opened automatically.

### Audio & text-to-speech

```
"Say 'Hello, welcome to my presentation' in a British female voice"
"Read this paragraph aloud: [paste text]"
"Generate audio of someone narrating this story"
"Convert this script to speech"
```

Uses Kokoro-82M by default (fast, high quality, ~1 GB). Output is saved as `.wav`
and played back automatically with `afplay`.

Available voices: `af_heart` (warm female), `af_bella`, `am_adam` (male), `bf_emma`
(British female), `bm_george` (British male).

### Vision / image analysis

```
"Describe what's in this image" [attach a photo]
"What text can you read in this screenshot?" [attach screenshot]
"Analyze this chart and explain the trends" [attach chart image]
"What objects are in the foreground of this photo?" [attach photo]
```

Claude selects a vision model (LLaVA or Qwen2-VL) based on your RAM, loads it locally,
and returns a description. The image never leaves your machine.

### Pipelines

```
"Generate an image of a mountain lake, then describe it back to me"
"Create a picture of a robot chef, then tell me what ingredients it's cooking with"
```

Claude chains image generation (mflux) → vision analysis (mlx-vlm) automatically.

---

## Model Selection & RAM Guide

The skill always picks the best model your RAM can safely hold. Here's the full map:

### Image generation (mflux / Flux)

| Your RAM | Model | Quantization | Steps | Approx. time |
|----------|-------|-------------|-------|--------------|
| 8 GB | Flux.1-schnell | 4-bit | 4 | ~45 sec |
| 8–15 GB | Flux.1-schnell | 8-bit | 4 | ~30 sec |
| 16–31 GB | Flux.1-dev | 8-bit | 20 | ~3 min |
| 32+ GB | Flux.1-dev | 8-bit | 20–50 | ~2–5 min |
| 64+ GB | Flux.1-dev | none (full) | 20–50 | Best quality |

**First run takes longer** because the model downloads from Hugging Face.
Flux.1-schnell is ~34 GB, Flux.1-dev is ~34 GB. Subsequent runs load from cache.

### Audio / TTS (mlx-audio)

| Model | RAM | Speed | Quality | Notes |
|-------|-----|-------|---------|-------|
| Kokoro-82M | ~1 GB | Fast | ⭐⭐⭐⭐ | Default |
| Bark-small | ~2 GB | Slower | ⭐⭐⭐⭐⭐ | More expressive |

TTS models are small — they work on any supported Mac.

### Vision / VLM (mlx-vlm)

| Your RAM | Model | Capability |
|----------|-------|-----------|
| 8–15 GB | LLaVA-1.5-7B-4bit | General image Q&A |
| 16–31 GB | Qwen2-VL-7B-Instruct-4bit | Excellent OCR, detail |
| 64+ GB | Qwen2-VL-72B-Instruct-4bit | Near-GPT4V quality |

---

## LAN Cluster Setup

A LAN cluster pools the RAM of multiple Macs so you can run models far larger than
any single machine could hold. This is optional — the skill works perfectly on a single
Mac — but it's a significant upgrade if you have spare Apple Silicon hardware.

### How distributed inference works

MLX uses **tensor parallelism**: the model's weight matrices are split across all nodes.
On each forward pass, every node computes its shard and then the results are merged via
collective operations (`all_reduce`) over the network. The ring backend uses plain TCP/IP,
so it works on any LAN — no special hardware required.

You run `mlx.launch` on the **controller node** (your main Mac). It SSHes into all other
nodes, starts the Python process on each, coordinates execution, and prints output back
to your terminal. From your perspective it looks like one inference run.

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
   (you run this here)
```

> **Important:** Distributed inference currently works best for **LLM inference** (large
> language models via `mlx-lm`). Image generation (mflux) and vision models (mlx-vlm)
> don't yet support multi-node sharding — for those workloads, the single node with the
> most RAM is used.

### Hardware recommendations

| Config | Total RAM | Best for |
|--------|-----------|----------|
| 2× Mac Mini M4 (16 GB) | 32 GB | Llama-3.1-8B full precision, 14B-4bit |
| 2× Mac Mini M4 (24 GB) | 48 GB | Llama-3.1-70B-4bit (tight) |
| 4× Mac Mini M4 (16 GB) | 64 GB | Llama-3.1-70B-4bit (comfortable) |
| 4× Mac Mini M4 (24 GB) | 96 GB | DeepSeek-R1-4bit, 70B full precision |
| 4× Mac Studio M2 Ultra (192 GB) | 768 GB | Literally anything |

**Network**: Wired gigabit Ethernet is strongly preferred. The ring backend sends large
tensors between nodes on every forward pass — a 1 Gbps link is workable, 2.5 Gbps or
10 Gbps is noticeably better. Wi-Fi works but adds latency.

**Uniformity helps but isn't required**: MLX handles heterogeneous RAM across nodes.
A cluster with one 32 GB and two 16 GB Macs works fine.

---

### Step 1 — Prepare every node

Do this on **every Mac in the cluster**, including the controller.

**a) Install the same Python version on all nodes, at the same path**

This is the most common source of cluster failures. All nodes must use the **exact same
Python binary path**.

```bash
# Check your Python path
which python3
# e.g. /usr/local/bin/python3  or  /opt/homebrew/bin/python3

# Install Python via Homebrew (recommended for consistency)
brew install python@3.11
```

Write down the path — you'll need it when generating the hostfile.

**b) Install MLX packages on every node**

```bash
pip install mlx mlx-lm huggingface_hub
```

For image/vision nodes (if you want those capabilities distributed in future):
```bash
pip install mflux mlx-vlm mlx-audio soundfile pillow
```

**c) Verify MLX works on each node**

```bash
python3 -c "import mlx; import mlx_lm; print('Node ready:', mlx.__version__)"
```

**d) Make sure hostnames resolve**

Macs advertise themselves via mDNS (Bonjour) as `<computername>.local`. Verify from
the controller:

```bash
ping -c 1 mac-mini-2.local
ping -c 1 mac-mini-3.local
```

If ping fails, either use IP addresses instead, or check that all Macs are on the same
subnet and that the firewall isn't blocking mDNS.

---

### Step 2 — Enable Remote Login

On **every non-controller node** (and optionally the controller too for symmetry):

1. Open **System Settings**
2. Go to **General → Sharing**
3. Enable **Remote Login**
4. Set **Allow access** to either "All users" or add your specific user account

From Terminal you can do the same:

```bash
# Enable Remote Login (run on each node)
sudo systemsetup -setremotelogin on

# Verify it's on
sudo systemsetup -getremotelogin
# Should print: Remote Login: On
```

---

### Step 3 — Set up passwordless SSH

`mlx.launch` SSHes into nodes automatically. It can't prompt for passwords, so you
need key-based auth configured beforehand.

Run all of this **on the controller node**:

```bash
# 1. Generate a dedicated SSH key for the cluster (skip if you already have one)
ssh-keygen -t ed25519 -C "mlx-cluster" -f ~/.ssh/id_mlx -N ""

# 2. Add it to your SSH agent so it's used automatically
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_mlx

# 3. Copy the public key to every node (enter password once per node)
ssh-copy-id -i ~/.ssh/id_mlx.pub your-username@mac-mini-2.local
ssh-copy-id -i ~/.ssh/id_mlx.pub your-username@mac-mini-3.local
ssh-copy-id -i ~/.ssh/id_mlx.pub your-username@mac-mini-4.local

# 4. Verify — these should NOT prompt for a password
ssh mac-mini-2.local "echo 'Node 2: SSH OK'"
ssh mac-mini-3.local "echo 'Node 3: SSH OK'"
ssh mac-mini-4.local "echo 'Node 4: SSH OK'"
```

To make the key persistent across reboots, add this to `~/.ssh/config`:

```
Host *.local
    IdentityFile ~/.ssh/id_mlx
    StrictHostKeyChecking no
    ServerAliveInterval 30
```

---

### Step 4 — Generate the hostfile

`mlx.launch` needs a JSON file describing all cluster nodes. The easiest way is the
built-in MLX config tool:

```bash
mlx.distributed_config \
  --backend ring \
  --over ethernet \
  --hosts mac-mini-1.local,mac-mini-2.local,mac-mini-3.local,mac-mini-4.local \
  --output ~/.mlx-cluster.json
```

Or generate it manually:

```json
[
  { "ssh": "192.168.1.10", "ips": ["192.168.1.10"], "python": "/usr/local/bin/python3" },
  { "ssh": "192.168.1.11", "ips": ["192.168.1.11"], "python": "/usr/local/bin/python3" },
  { "ssh": "192.168.1.12", "ips": ["192.168.1.12"], "python": "/usr/local/bin/python3" }
]
```

> **Use IPs not hostnames in the JSON** if you're on a network where `.local` resolution
> is unreliable. Use `arp -a` or check your router's DHCP table to find IPs.

The skill can also **auto-discover nodes** — when you ask Claude something that triggers
distributed inference, it will scan your subnet for Macs with MLX installed and offer to
build the hostfile for you.

---

### Step 5 — Verify cluster readiness

Run the cluster check script from the controller. Save it as `check_cluster.py` and run
it before the first distributed inference:

```python
import subprocess, json

HOSTS = [
    "mac-mini-2.local",
    "mac-mini-3.local",
    "mac-mini-4.local",
]
PYTHON_PATH = "/usr/local/bin/python3"  # must match all nodes

def check_node(host):
    results = {"host": host}

    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, "echo ok"],
        capture_output=True, text=True
    )
    results["ssh"] = r.returncode == 0
    if not results["ssh"]:
        results["error"] = r.stderr.strip()
        return results

    r = subprocess.run(["ssh", host, "uname -m"], capture_output=True, text=True)
    results["apple_silicon"] = r.stdout.strip() == "arm64"

    r = subprocess.run(
        ["ssh", host, f"{PYTHON_PATH} -c 'import mlx; print(mlx.__version__)'"],
        capture_output=True, text=True
    )
    results["mlx"] = r.stdout.strip() if r.returncode == 0 else "MISSING"

    r = subprocess.run(
        ["ssh", host, f"{PYTHON_PATH} -c 'import mlx_lm; print(mlx_lm.__version__)'"],
        capture_output=True, text=True
    )
    results["mlx_lm"] = r.stdout.strip() if r.returncode == 0 else "MISSING"

    r = subprocess.run(["ssh", host, "sysctl -n hw.memsize"], capture_output=True, text=True)
    if r.returncode == 0:
        results["ram_gb"] = round(int(r.stdout.strip()) / 1e9, 1)

    r = subprocess.run(["ssh", host, "ipconfig getifaddr en0"], capture_output=True, text=True)
    results["ip"] = r.stdout.strip()

    return results

total_ram = 0
all_ok = True
print(f"{'Host':<25} {'SSH':>5} {'Silicon':>8} {'MLX':>10} {'mlx-lm':>10} {'RAM':>8} {'IP'}")
print("─" * 85)
for host in HOSTS:
    r = check_node(host)
    ok = r.get("ssh") and r.get("apple_silicon") and r.get("mlx") != "MISSING" and r.get("mlx_lm") != "MISSING"
    ram = r.get("ram_gb", 0)
    total_ram += ram
    status = "✅" if ok else "❌"
    print(f"{status} {host:<23} {str(r.get('ssh')):>5} {str(r.get('apple_silicon')):>8} "
          f"{str(r.get('mlx')):>10} {str(r.get('mlx_lm')):>10} "
          f"{str(ram)+'GB':>8} {r.get('ip', 'unknown')}")
    if not ok:
        all_ok = False
        if r.get("error"):
            print(f"   └─ {r['error']}")

print("─" * 85)
print(f"Pooled RAM: {total_ram} GB")
print(f"Cluster: {'READY ✅' if all_ok else 'NOT READY ❌ — fix issues above'}")
```

```bash
python3 check_cluster.py
```

Expected healthy output:
```
Host                      SSH  Silicon        MLX     mlx-lm      RAM IP
─────────────────────────────────────────────────────────────────────────────────────
✅ mac-mini-2.local        True     True     0.30.6     0.22.1    24.0GB 192.168.1.11
✅ mac-mini-3.local        True     True     0.30.6     0.22.1    24.0GB 192.168.1.12
✅ mac-mini-4.local        True     True     0.30.6     0.22.1    16.0GB 192.168.1.13
─────────────────────────────────────────────────────────────────────────────────────
Pooled RAM: 64.0 GB
Cluster: READY ✅
```

---

### Step 6 — Run distributed inference

Once the cluster is verified, run from the **controller node**:

```bash
# Basic generation across all nodes
mlx.launch \
  --verbose \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  -- \
  /usr/local/bin/python3 -m mlx_lm.generate \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit \
    --prompt "Explain the theory of relativity in simple terms" \
    --max-tokens 500

# Interactive chat REPL
mlx.launch \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  -- \
  /usr/local/bin/python3 -m mlx_lm.chat \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit
```

For better LAN throughput, add the performance hint:

```bash
mlx.launch \
  --backend ring \
  --hostfile ~/.mlx-cluster.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  -- \
  /usr/local/bin/python3 -m mlx_lm.chat \
    --model mlx-community/Llama-3.1-70B-Instruct-4bit
```

**From OpenClaw**, you can also just ask naturally:

```
"Use the cluster to run Llama 70B and explain quantum entanglement"
"Run DeepSeek-R1 across all my Macs and solve this math problem: [problem]"
```

The skill will detect the hostfile, verify nodes, and launch automatically.

---

### Cluster RAM & model guide

| Cluster | Pooled RAM | Unlocks |
|---------|-----------|---------|
| 2× 16 GB | 32 GB | Llama-3.1-8B (full), Qwen2.5-14B-4bit |
| 2× 24 GB | 48 GB | Llama-3.1-70B-4bit (tight) |
| 2× 32 GB | 64 GB | Llama-3.1-70B-4bit (comfortable) |
| 4× 16 GB | 64 GB | Llama-3.1-70B-4bit, DeepSeek-V3-4bit |
| 4× 24 GB | 96 GB | DeepSeek-R1-4bit, Llama-3.1-70B full precision |
| 4× 32 GB | 128 GB | Any current open model at full precision |
| 8× 16 GB | 128 GB | Same as above |

**Best distributed models** (tested, tensor-parallel friendly):

- `mlx-community/Llama-3.1-70B-Instruct-4bit` — 40 GB, flagship general model
- `mlx-community/DeepSeek-R1-0528-4bit` — outstanding reasoning, very large
- `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` — MoE architecture distributes efficiently
- `mlx-community/Qwen2.5-72B-Instruct-4bit` — excellent multilingual + coding

---

## Network & Firewall

By default, macOS's firewall may block the ports MLX uses for inter-node communication.
If nodes hang waiting for each other, allow the relevant ports:

```bash
# Allow incoming connections on ports used by MLX ring backend (run on all nodes)
/usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/bin/python3
/usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/bin/python3
```

Or temporarily disable the application firewall for testing:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
```

Re-enable after testing:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
```

**Ports used by MLX ring backend**: The ring backend uses ephemeral ports negotiated at
startup, typically in the 5000–60000 range. If you're running a strict firewall policy,
the easiest solution is to whitelist Python itself rather than individual ports.

---

## Troubleshooting

### Single Mac issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `platform.machine()` returns `x86_64` | Intel Mac | MLX not supported — use cloud APIs |
| `RuntimeError: Out of memory` | Model too large | Use higher quantization (`quantize=4`) or smaller model |
| Process killed silently (SIGKILL) | macOS OOM killer | Same as above — reduce model size |
| `403 Forbidden` downloading model | HF auth needed | `huggingface-cli login` |
| `Repository not found` | Model ID typo, or HF auth needed | Check model page on HF |
| Very slow generation (>10 min) | No quantization | Add `quantize=8` |
| Black or corrupted image | Stale mflux version | `pip install -U mflux` |
| VLM output is garbled | Wrong prompt template | Ensure `apply_chat_template` is used |
| First run takes 20–30 min | Model downloading | Normal — models are 10–35 GB |

### Cluster issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` on port 22 | Remote Login not enabled | System Settings → Sharing → Remote Login |
| `Permission denied (publickey)` | SSH key not deployed | `ssh-copy-id user@node` |
| `python3: command not found` | Path mismatch across nodes | Use full absolute path in hostfile |
| Rank 0 starts, others hang | Firewall blocking ports | Allow Python through macOS firewall (see above) |
| `mx.distributed.init()` returns `size=1` | `mlx.launch` not used | Must use `mlx.launch`, not `python3` directly |
| `.local` hostname resolution fails | mDNS/Bonjour issue | Use IP addresses in hostfile instead |
| Slow generation despite multiple nodes | Wi-Fi bottleneck | Switch to wired Ethernet |
| `KeyError: 'domain_uuid_key'` | TB backend mismatch | Use `--backend ring --over ethernet` |
| Nodes get unequal load | Heterogeneous RAM | Normal — MLX handles this automatically |
| Model download happens on every node | No shared cache | See below — share HF cache via NFS or pre-download |

### Speed up first run: pre-download models

If you have multiple nodes, you can avoid downloading the same model 4 times by sharing
the Hugging Face cache directory over NFS, or by pre-downloading on one node and
`rsync`-ing to the others:

```bash
# Pre-download on controller
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Llama-3.1-70B-Instruct-4bit')
"

# Sync to other nodes (repeat for each)
rsync -avz --progress \
  ~/.cache/huggingface/hub/models--mlx-community--Llama-3.1-70B-Instruct-4bit \
  your-username@mac-mini-2.local:~/.cache/huggingface/hub/
```

---

## Extending the Skill

The skill is designed to be extended as new MLX packages appear. To add a new model
or capability, edit `SKILL.md`:

1. **Add the package** to the `ensure(...)` call in Step 3
2. **Add a new entry** to the category table in Step 1
3. **Add RAM requirements** to Step 2
4. **Add usage code** to Step 5 under a new subsection
5. **Add to the tracked projects table** at the bottom of the skill
6. **Add any new error patterns** to the Error Handling section

### Currently supported MLX projects

| Project | PyPI Package | HF Namespace | Status |
|---------|-------------|--------------|--------|
| Flux image generation | `mflux` | `black-forest-labs` | ✅ Stable |
| Audio / TTS | `mlx-audio` | `prince-canuma` | ✅ Stable |
| Vision / VLM | `mlx-vlm` | `mlx-community` | ✅ Stable |
| LLM inference (distributed) | `mlx-lm` | `mlx-community` | ✅ Stable |
| Whisper STT | `mlx-whisper` | `mlx-community` | ✅ Stable |
| Stable Audio / Music | via `mlx-audio` | `stabilityai` | 🧪 Experimental |
| SDXL | `mlx-stable-diffusion` | `apple` | 🧪 Experimental |

---

## File Layout

```
/Users/clawd/clawd/skills/mlx/
├── SKILL.md          ← The skill itself (read by OpenClaw/Claude)
└── README.md         ← This file

~/.cache/huggingface/  ← Model cache (auto-managed by HF hub)
~/Desktop/mlx-outputs/ ← Default output directory for generated files
~/.mlx-cluster.json    ← Cluster hostfile (created during cluster setup)
~/.ssh/id_mlx          ← SSH key for cluster auth (created during cluster setup)
```

---

## Quick Reference Card

```bash
# ── Single Mac ─────────────────────────────────────────────────────────────

# Generate image (fast)
mflux-generate --model flux-schnell --prompt "your prompt" \
  --steps 4 --quantize 8 --output ~/Desktop/mlx-outputs/out.png

# Generate image (quality)
mflux-generate --model flux-dev --prompt "your prompt" \
  --steps 20 --quantize 8 --output ~/Desktop/mlx-outputs/out.png

# Text to speech
python3 -c "
from mlx_audio.tts.generate import generate_audio; import soundfile as sf
audio, sr = generate_audio('Hello world', model='prince-canuma/Kokoro-82M', voice='af_heart')
sf.write('out.wav', audio, sr)
import subprocess; subprocess.Popen(['afplay', 'out.wav'])
"

# Describe an image
python3 -c "
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
m, p = load('mlx-community/llava-1.5-7b-4bit')
c = load_config('mlx-community/llava-1.5-7b-4bit')
f = apply_chat_template(p, c, 'Describe this image.', num_images=1)
print(generate(m, p, 'photo.jpg', f, verbose=False))
"

# ── Cluster ────────────────────────────────────────────────────────────────

# Check cluster health
python3 check_cluster.py

# Generate hostfile
mlx.distributed_config --backend ring --over ethernet \
  --hosts node1.local,node2.local,node3.local \
  --output ~/.mlx-cluster.json

# Run distributed LLM
mlx.launch --backend ring --hostfile ~/.mlx-cluster.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  -- /usr/local/bin/python3 -m mlx_lm.chat \
     --model mlx-community/Llama-3.1-70B-Instruct-4bit
```

---

*Skill version: 1.0 · Compatible with MLX 0.20+ · Requires macOS 13.5+ on Apple Silicon*
