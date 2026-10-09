#!/usr/bin/env bash
# Read-only environment report for running Jeff on an Ascend (910B) host.
#
#   bash deploy/ascend-check.sh
#
# It changes nothing on the machine. Run it before deploying and paste the
# output when asking for help -- it answers the questions that decide whether
# the CPU install will go smoothly: architecture, glibc, CANN paths, whether
# PYTHONPATH is contaminated with torch_npu, and how many instances to run.

set -uo pipefail

BOLD='\033[1m'; DIM='\033[2m'; R='\033[0m'
CY='\033[1;36m'; YL='\033[1;33m'; GR='\033[1;32m'; RD='\033[1;31m'
h()  { printf "\n${BOLD}%s${R}\n" "$*"; }
kv() { printf "  %-26s %s\n" "$1" "$2"; }
ok() { printf "  ${GR}%s${R}\n" "$*"; }
wn() { printf "  ${YL}%s${R}\n" "$*"; }
er() { printf "  ${RD}%s${R}\n" "$*"; }

h "1. System"
ARCH="$(uname -m)"
kv "architecture" "$ARCH"
[ -f /etc/os-release ] && kv "os" "$(. /etc/os-release && echo "$PRETTY_NAME")"
kv "kernel" "$(uname -r)"
if command -v ldd >/dev/null 2>&1; then
  GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
  kv "glibc" "$GLIBC"
  # manylinux_2_28 wheels need glibc >= 2.28
  if [ -n "$GLIBC" ] && [ "$(printf '%s\n' 2.28 "$GLIBC" | sort -V | head -1)" = "2.28" ]; then
    ok "glibc >= 2.28, manylinux_2_28 wheels will load"
  else
    er "glibc < 2.28: the torch CPU wheels will not load. Use Ubuntu 20.04+ / openEuler 22.03+ / Kylin V10 SP2+."
  fi
fi
case "$ARCH" in
  x86_64|amd64) ok "x86_64: all wheels available" ;;
  aarch64)      ok "aarch64 (Kunpeng): torch/torchvision CPU wheels available" ;;
  *)            er "unsupported architecture $ARCH" ;;
esac

h "2. CPU and memory"
CPUS="$(nproc 2>/dev/null || echo '?')"
kv "logical cores" "$CPUS"
if [ -r /proc/cpuinfo ]; then
  MODEL="$(grep -m1 -E '^(model name|Hardware)' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
  kv "cpu" "$MODEL"
fi
free -g >/dev/null 2>&1 && kv "memory" "$(free -g | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"

h "3. NPU and CANN"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info 2>/dev/null | head -20
elif [ -x /usr/local/Ascend/driver/tools/npu-smi ]; then
  /usr/local/Ascend/driver/tools/npu-smi info 2>/dev/null | head -20
else
  wn "npu-smi not found (driver not installed, or PATH missing /usr/local/Ascend/driver/tools)"
fi

ASCEND_BASE=""
for p in /usr/local/Ascend "$HOME/Ascend"; do
  [ -d "$p" ] && ASCEND_BASE="$p" && break
done
if [ -n "$ASCEND_BASE" ]; then
  kv "Ascend base" "$ASCEND_BASE"
  # CANN 8.5+ layout is <base>/cann/<arch>-linux; older is <base>/ascend-toolkit/latest/<arch>-linux
  for f in "$ASCEND_BASE/cann/$ARCH-linux/ascend_toolkit_install.info" \
           "$ASCEND_BASE/ascend-toolkit/latest/$ARCH-linux/ascend_toolkit_install.info" \
           "$ASCEND_BASE/ascend-toolkit/latest/version.info"; do
    if [ -f "$f" ]; then
      kv "CANN" "$(grep -iE '^(version|Version)=' "$f" | head -1 | cut -d= -f2-)"
      kv "CANN info file" "$f"
      break
    fi
  done
else
  wn "no Ascend toolkit found under /usr/local/Ascend or ~/Ascend"
fi

h "4. Environment contamination (the part that breaks CPU mode)"
if [ -n "${PYTHONPATH:-}" ]; then
  wn "PYTHONPATH is set:"
  printf '%s\n' "$PYTHONPATH" | tr ':' '\n' | sed 's/^/    /'
  if printf '%s' "$PYTHONPATH" | grep -qiE 'ascend|cann'; then
    er "PYTHONPATH contains Ascend/CANN paths -- torch_npu there can hijack torch.cuda.is_available()"
    er "and make Jeff try to run on a CUDA device. Strip it before starting jeff-serve."
  fi
else
  ok "PYTHONPATH is empty"
fi

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  kv "LD_LIBRARY_PATH" "set ($(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -ciE 'ascend|cann') Ascend entries)"
  printf '%s' "$LD_LIBRARY_PATH" | grep -qiE 'ascend' && wn "Ascend libs on LD_LIBRARY_PATH are fine, they do not affect torch device selection"
else
  kv "LD_LIBRARY_PATH" "empty"
fi

for v in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH; do
  val="${!v:-}"
  [ -n "$val" ] && kv "$v" "$val"
done

h "5. Python"
if command -v python3 >/dev/null 2>&1; then
  kv "python3" "$(python3 --version 2>&1)"
else
  er "python3 not found"
fi
command -v uv >/dev/null 2>&1 && ok "uv: $(uv --version)" || wn "uv not installed (the installer will add it)"
# does importing torch pull in torch_npu?
python3 - <<'PY' 2>/dev/null || true
try:
    import torch
    print(f"  {'torch':<26} {torch.__version__}  cuda.is_available()={torch.cuda.is_available()}")
    try:
        import torch_npu  # noqa: F401
        print("  \033[1;33mtorch_npu is importable -- do NOT let it into the jeff venv\033[0m")
    except ImportError:
        print("  \033[1;32mtorch_npu not importable (good)\033[0m")
except ImportError:
    print("  torch not installed in system python yet")
PY

h "6. Disk and network"
kv "free space (cwd)" "$(df -h . 2>/dev/null | awk 'NR==2{print $4}')"
for u in "https://mirrors.aliyun.com/pytorch-wheels/cpu/" "https://mirrors.aliyun.com/pypi/simple/" "https://hf-mirror.com"; do
  code="$(curl -s -o /dev/null -m 12 -w '%{http_code}' -L "$u" 2>/dev/null || echo fail)"
  case "$code" in
    200|206|301|302) ok "$code  $u" ;;
    *) er "$code  $u -- unreachable, check proxy/DNS" ;;
  esac
done

h "7. Recommended settings"
if [ "$CPUS" != "?" ]; then
  if   [ "$CPUS" -ge 96 ]; then T=16
  elif [ "$CPUS" -ge 48 ]; then T=16
  elif [ "$CPUS" -ge 24 ]; then T=8
  elif [ "$CPUS" -ge 12 ]; then T=8
  else T=4; fi
  INST=$((CPUS / T)); [ "$INST" -gt 8 ] && INST=8
  kv "OMP_NUM_THREADS" "$T   (small models slow down past ~16 threads)"
  kv "instances" "$INST   (jeff-serve is serial: one request at a time per instance)"
  echo
  echo "  export OMP_NUM_THREADS=$T OMP_PROC_BIND=false MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T"
  echo "  export JEFF_DEVICE=cpu JEFF_QUEUE_MS=2000"
  echo "  # start $INST instances on ports 8765..$((8764+INST)) and put nginx in front"
fi

printf "\n${DIM}Report complete. Nothing was modified.${R}\n"
