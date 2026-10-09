#!/usr/bin/env bash
# Jeff CPU-only installer for Linux (x86_64 / aarch64).
# Target: Ascend 910B servers, CPU mode -- see DEPLOY_NOTES_LINUX.md.
#
# Usage:
#   bash deploy/install-cpu.sh              # install + download weights + start
#   SKIP_START=1 bash deploy/install-cpu.sh # install only
#
# Tunables (env): JEFF_DIR PORT JEFF_HOST PY_VERSION REPO PYPI_INDEX TORCH_MIRROR
#                 HF_ENDPOINT CKPT_REPO SKIP_START

set -euo pipefail

JEFF_DIR="${JEFF_DIR:-$HOME/jeff}"
PORT="${PORT:-8765}"
JEFF_HOST="${JEFF_HOST:-0.0.0.0}"
PY_VERSION="${PY_VERSION:-3.12}"
REPO="${REPO:-https://gh-proxy.com/https://github.com/ForeverAugust/jeff.git}"
PYPI_INDEX="${PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
TORCH_MIRROR="${TORCH_MIRROR:-https://mirrors.aliyun.com/pytorch-wheels/cpu}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CKPT_REPO="${CKPT_REPO:-jeff-legacy/Jeff-Qwen3.5-0.8B}"
SKIP_START="${SKIP_START:-0}"
CKPT_DIR="checkpoints/jeff-0.8b"

log() { printf '\033[1;36m[jeff]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[jeff]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[jeff]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. platform
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  WHEEL_ARCH="x86_64"  ;;
  aarch64) WHEEL_ARCH="aarch64" ;;
  *) die "Unsupported architecture: $ARCH (need x86_64 or aarch64)" ;;
esac
PY_TAG="cp${PY_VERSION//./}"
log "platform: $ARCH, python $PY_VERSION"

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v git  >/dev/null 2>&1 || die "git is required"

# Ascend hosts often export a PYTHONPATH pointing at CANN's python packages.
# torch_npu there can hijack torch.cuda.is_available() and break CPU mode.
if [ -n "${PYTHONPATH:-}" ]; then
  warn "PYTHONPATH is set; unsetting it for this shell to avoid torch_npu contamination"
  unset PYTHONPATH
fi

# ---------------------------------------------------------------- 1. uv
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; add \$HOME/.local/bin and re-run"
else
  log "uv already present: $(uv --version)"
fi

# ---------------------------------------------------------------- 2. source
if [ ! -d "$JEFF_DIR/.git" ]; then
  log "cloning $REPO -> $JEFF_DIR"
  git clone --depth 1 "$REPO" "$JEFF_DIR"
else
  log "using existing checkout: $JEFF_DIR"
fi
cd "$JEFF_DIR"

# ---------------------------------------------------------------- 3. venv
log "creating venv (python $PY_VERSION)"
uv python install "$PY_VERSION" >/dev/null 2>&1 || true
uv venv --python "$PY_VERSION"

# ---------------------------------------------------------------- 4. CPU torch
# Linux PyPI wheels for torch are the CUDA build (~4GB of nvidia-* deps).
# On a machine without an NVIDIA card that is pure waste, so pin the +cpu wheels.
mkdir -p wheels
for spec in "torch:2.14.0" "torchvision:0.29.0"; do
  pkg="${spec%%:*}"
  ver="${spec##*:}"
  whl="$pkg-$ver+cpu-$PY_TAG-$PY_TAG-manylinux_2_28_${WHEEL_ARCH}.whl"
  if [ ! -f "wheels/$whl" ]; then
    log "downloading $whl"
    curl -fL --retry 3 --retry-delay 2 -o "wheels/$whl" "$TORCH_MIRROR/$whl" \
      || die "Failed to download $whl from $TORCH_MIRROR"
  else
    log "$whl already cached"
  fi
done
log "installing CPU-only torch"
uv pip install wheels/*.whl

# ---------------------------------------------------------------- 5. remaining deps
log "installing remaining dependencies (skipping lockfile torch)"
if ! UV_INDEX_URL="$PYPI_INDEX" uv sync --no-default-groups \
      --no-install-package torch --no-install-package torchvision; then
  warn "uv sync failed (old uv without --no-install-package?); falling back to explicit install"
  uv pip install --index-url "$PYPI_INDEX" \
    "transformers==5.17.0" "pillow==12.3.0" "fastapi==0.141.1" "uvicorn==0.52.4" \
    "safetensors==0.8.0" "numpy==2.5.3" "huggingface-hub==1.31.0"
  uv pip install --no-deps -e .
fi

# sanity: torch must be the CPU build
uv run --no-default-groups python - <<'PY' || die "torch did not install as a CPU build"
import torch, sys
v = torch.__version__
print(f"[jeff] torch {v}, cuda available: {torch.cuda.is_available()}")
if "cpu" not in v and torch.cuda.is_available():
    sys.exit("got a CUDA torch build; this host has no usable NVIDIA device")
PY

# ---------------------------------------------------------------- 6. weights
if [ ! -f "$CKPT_DIR/model.safetensors" ]; then
  log "downloading weights from $CKPT_REPO (1.7GB via $HF_ENDPOINT)"
  mkdir -p "$CKPT_DIR"
  if ! HF_ENDPOINT="$HF_ENDPOINT" uv run --no-default-groups \
        hf download "$CKPT_REPO" --local-dir "$CKPT_DIR"; then
    warn "hf CLI failed; falling back to curl"
    for f in config.json decision_config.json chat_template.jinja tokenizer.json \
             tokenizer_config.json processor_config.json model.safetensors readout.safetensors; do
      curl -fL -o "$CKPT_DIR/$f" "https://hf-mirror.com/$CKPT_REPO/resolve/main/$f" \
        || die "Failed to download $f"
    done
  fi
else
  log "weights already present"
fi

for f in config.json decision_config.json chat_template.jinja tokenizer.json \
         tokenizer_config.json processor_config.json model.safetensors readout.safetensors; do
  [ -f "$CKPT_DIR/$f" ] || die "Missing checkpoint file: $CKPT_DIR/$f"
done
log "checkpoint complete: $CKPT_DIR"

if [ "$SKIP_START" = "1" ]; then
  log "SKIP_START=1, not starting the server"
  log "start manually with:  JEFF_CHECKPOINT=$CKPT_DIR JEFF_DEVICE=cpu JEFF_HOST=$JEFF_HOST PORT=$PORT JEFF_QUEUE_MS=2000 .venv/bin/jeff-serve"
  exit 0
fi

# ---------------------------------------------------------------- 7. start
log "starting jeff-serve on $JEFF_HOST:$PORT"
JEFF_CHECKPOINT="$CKPT_DIR" \
JEFF_DEVICE=cpu \
JEFF_HOST="$JEFF_HOST" \
PORT="$PORT" \
JEFF_QUEUE_MS=2000 \
  nohup .venv/bin/jeff-serve > jeff-serve.log 2>&1 &

log "waiting for the model to load (cold start takes 20-60s)"
for _ in $(seq 1 60); do
  if curl -sf --noproxy '*' "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    log "ready: http://$JEFF_HOST:$PORT/v1/systemone"
    log "health:  curl -s http://127.0.0.1:$PORT/v1/models"
    log "probe:   curl -s http://127.0.0.1:$PORT/v1/systemone -H 'content-type: application/json' -d @test_request.json"
    log "logs:    tail -f $JEFF_DIR/jeff-serve.log"
    exit 0
  fi
  sleep 5
done

die "Server did not become ready in 5 minutes. Check $JEFF_DIR/jeff-serve.log"
