# Jeff Linux 部署笔记（CPU 模式，昇腾 910B 服务器）

在昇腾（Ascend 910B / Atlas 800 A2）服务器上以 **CPU 模式**跑 Jeff-Qwen3.5-0.8B 的完整记录。
NPU 加速另有一套方案（见第八节），这份笔记只解决「先跑起来、验证业务价值」。

配套文件：

| 文件 | 用途 |
|---|---|
| `deploy/ascend-check.sh` | 只读诊断：跑一遍看机器状态和昇腾特有的冲突（建议先跑这个） |
| `deploy/install-cpu.sh` | 一键部署（检测架构 → 装 CPU 版 torch → 下载权重 → 启动） |
| `deploy/jeff-cpu.service` | systemd 单元，常驻运行 |

通用 Linux（NVIDIA GPU / x86_64 / aarch64）部署指导见 `LINUX_DEPLOY_GUIDE.md`；Windows 版本见 `DEPLOY_NOTES.md`。

---

## 一、为什么是 CPU 模式

源码里 `JEFF_DEVICE` 白名单只有三个（`src/jeff/models.py:37`）：

```python
available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
```

写 `npu` 直接报 `ValueError`。昇腾上跑必须走 CPU，或者改代码（第八节）。好消息是**功能完全不受影响**，只是慢。

---

## 二、环境要求

| 项目 | 要求 | 备注 |
|---|---|---|
| 架构 | x86_64 或 aarch64（鲲鹏 920） | 脚本自动检测，两种都实测过 wheel 可用 |
| Python | 3.12 / 3.13 / 3.14 | 项目要求 `>=3.12`；脚本固定用 3.12（wheel 文件名最全） |
| 内存 | ≥ 8GB（模型 float32 约 3.2GB + 运行时） | 昇腾服务器通常 512GB+，不是问题 |
| 磁盘 | 权重 1.7GB + venv 约 2GB | CPU 版 torch 只占 ~200MB |
| 系统 | openEuler / Ubuntu 20.04+ / CentOS 8+ | glibc ≥ 2.28（manylinux_2_28 要求） |

---

## 三、一键部署

先跑诊断（只读，不改任何东西），把输出留着对照：

```bash
git clone https://gh-proxy.com/https://github.com/ForeverAugust/jeff.git jeff
cd jeff
bash deploy/ascend-check.sh
```

它会报出：架构与 glibc、CPU 核数与内存、`npu-smi` 与 CANN 版本、**PYTHONPATH 是否被 CANN 污染**、系统 Python 里有没有 `torch_npu`、三个镜像源的连通性，以及推荐的线程数和实例数。

然后安装：

```bash
bash deploy/install-cpu.sh
```

脚本会做六件事：检测架构 → 装 uv → 建 venv → 装 CPU 版 torch → 下载权重 → 启动并自检。

可选环境变量（不设就用默认值）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `JEFF_DIR` | `~/jeff` | 安装目录 |
| `PORT` | `8765` | 服务端口 |
| `JEFF_HOST` | `0.0.0.0` | 监听地址，脚本默认对外放开 |
| `PYPI_INDEX` | 阿里云 | PyPI 镜像 |
| `HF_ENDPOINT` | `hf-mirror.com` | 权重镜像 |

只装不启动：`SKIP_START=1 bash deploy/install-cpu.sh`

---

## 四、手动分步（想自己控制每一步时照这个来）

```bash
# 1. uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. 代码
git clone https://gh-proxy.com/https://github.com/ForeverAugust/jeff.git jeff && cd jeff

# 3. venv（固定 3.12）
uv python install 3.12
uv venv --python 3.12

# 4. CPU 版 torch —— 这一步是 Linux 和 Windows 最大的差别，见第五节
ARCH=$(uname -m)   # x86_64 / aarch64
BASE="https://mirrors.aliyun.com/pytorch-wheels/cpu"
curl -LO "$BASE/torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_${ARCH}.whl"
curl -LO "$BASE/torchvision-0.29.0+cpu-cp312-cp312-manylinux_2_28_${ARCH}.whl"
uv pip install ./*.whl

# 5. 其余依赖（跳过 lockfile 里的 CUDA 版 torch）
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  uv sync --no-default-groups --no-install-package torch --no-install-package torchvision

# 6. 权重
export HF_ENDPOINT=https://hf-mirror.com
uv run --no-default-groups hf download jeff-legacy/Jeff-Qwen3.5-0.8B \
  --local-dir checkpoints/jeff-0.8b

# 7. 启动
JEFF_CHECKPOINT=checkpoints/jeff-0.8b JEFF_DEVICE=cpu \
JEFF_HOST=0.0.0.0 PORT=8765 JEFF_QUEUE_MS=2000 \
  uv run --no-default-groups jeff-serve
```

权重目录核心是 `model.safetensors`（1,706,027,688 字节）和 `readout.safetensors`（决策头，255×1024，约 522KB），另需 `config.json`、`decision_config.json`、`chat_template.jinja`、`tokenizer.json`、`tokenizer_config.json`、`processor_config.json`。
`hf` CLI 若卡住，改用 curl 逐个下：

```bash
for f in config.json decision_config.json chat_template.jinja tokenizer.json \
         tokenizer_config.json processor_config.json model.safetensors readout.safetensors; do
  curl -L -o "checkpoints/jeff-0.8b/$f" \
    "https://hf-mirror.com/jeff-legacy/Jeff-Qwen3.5-0.8B/resolve/main/$f"
done
```

---

## 五、Linux 特有的四个坑

### 1. 默认会装 CUDA 版 torch（最容易中招）

Windows 上 PyPI 的 torch wheel 是 CPU 版，所以 `uv sync` 直接就能用。
**Linux 不是。** lockfile 里 torch 2.14.0 的 Linux 依赖长这样：

```
cuda-bindings / cuda-toolkit (cublas cudart cufft cufile cupti curand cusolver cusparse nvjitlink nvrtc nvtx)
nvidia-cudnn-cu13 / nvidia-cusparselt-cu13 / nvidia-nccl-cu13 / nvidia-nvshmem-cu13 / triton
```

直接 `uv sync` 会拖下来 **4GB+ 的 CUDA 全家桶**，在没有 NVIDIA 卡的昇腾服务器上纯属浪费。
必须先手工装 CPU 版（`torch-2.14.0+cpu`，200MB 左右），再用 `--no-install-package` 跳过。

版本兼容性不用担心：`2.14.0+cpu` 满足 `torch==2.14.0`（PEP 440 忽略 local version 段）。

### 2. 默认只监听 127.0.0.1

`src/jeff/server.py` 最后一行：

```python
uvicorn.run("jeff.server:app", host=os.getenv("JEFF_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
```

服务器上必须显式 `JEFF_HOST=0.0.0.0`，否则外部机器连不上（现象是本机 curl 通、别的机器 connection refused）。

### 3. 一次只处理一个请求 → HTTP 529

服务端是单线程串行推理。请求撞上模型忙时，等待 `JEFF_QUEUE_MS`（**默认 0**）后返回 **529 + `Retry-After: 1`**。

CPU 模式下单请求 ~1s，这个限制比 GPU 上痛得多。必须二选一：

- 设 `JEFF_QUEUE_MS=2000` 让请求排队（超出才 529）
- 客户端对 529 做退避重试

### 4. 昇腾环境的 torch_npu 污染

如果系统里装了 `torch_npu`，且通过 `PYTHONPATH` 或 sitecustomize 注入，它可能劫持 `torch.cuda.is_available()` 返回 True —— 那样 Jeff 会尝试把张量搬到 "cuda"，在 NPU 上直接崩。

**对策**：jeff 的 venv 里不要装 torch_npu；始终显式 `JEFF_DEVICE=cpu`；启动前 `unset PYTHONPATH` 或确认它不含 CANN 的 python 路径。

---

## 六、昇腾（910B）专项

### 6.1 CANN 环境变量污染——昇腾头号坑

昇腾装完 CANN 后，文档都会让你把这段写进 `.bashrc`：

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
source $ASCEND_TOOLKIT_HOME/set_env.sh
export PYTHONPATH=$ASCEND_TOOLKIT_HOME/python/site-packages:$PYTHONPATH   # ← 问题在这行
```

最后一行把 CANN 的 Python 包（含 `torch_npu`）暴露给了**所有** Python 进程。`torch_npu` 会劫持 `torch.cuda.is_available()`，后果是：

1. Jeff 的 `device_from_environment()` 看到 `cuda: True`，选了 cuda 设备
2. 把张量往 "cuda" 上搬 → NPU 上没有 CUDA runtime → 崩或静默错误

**对策**（`install-cpu.sh` 已自动做）：只剥离 PYTHONPATH 里含 `ascend`/`cann` 的条目，其余保留；同时显式 `JEFF_DEVICE=cpu`。

| 变量 | 是否危险 | 说明 |
|---|---|---|
| `PYTHONPATH` 含 Ascend 路径 | ⚠️ 危险 | `torch_npu` 劫持设备检测 |
| `LD_LIBRARY_PATH` 含 CANN lib | ✅ 无害 | 只是动态库搜索路径，不影响设备选择 |
| `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` | ✅ 无害 | 纯路径变量 |

自检：

```bash
python3 -c "import torch_npu" 2>&1 | tail -1   # 期望：No module named 'torch_npu'
```

装完 Jeff 后在它的 venv 里再确认一次：

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出：2.14.0+cpu False
```

### 6.2 线程调优：核多不等于快

昇腾服务器常见 64–128 核（鲲鹏 920）。但 Jeff 是 0.8B 小模型的**短串行图**，线程开多了调度开销反超计算收益 —— 实测规律是 8–16 线程见顶，再多就变慢。

| 逻辑核数 | `OMP_NUM_THREADS`（每实例） | 建议实例数 |
|---|---|---|
| ≥ 96 | 16 | 6–8 |
| 48–95 | 16 | 3–4 |
| 24–47 | 8 | 3–4 |
| 12–23 | 8 | 2 |
| < 12 | 4 | 1 |

`install-cpu.sh` 会按核数自动设置这几个变量：

```bash
export OMP_NUM_THREADS=$T OMP_PROC_BIND=false MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
```

**多实例优于多线程**：jeff-serve 单实例串行，一个 16 线程实例 ≈ 1 QPS；4 个实例各自 16 线程 ≈ 4 QPS。

### 6.3 鲲鹏（aarch64）上不要做的事

| 做法 | 结论 |
|---|---|
| 指望 SVE 加速 / 设 `TORCH_ARM_SVE=1` | ❌ 鲲鹏 920 是 ARMv8.2，**不支持 SVE**，网上流传的这个变量无效 |
| 把 `torch_npu` 装进 jeff 的 venv | ❌ 见 6.1 |
| CPU 模式改 bfloat16 | ❌ Jeff 在 CPU 下自动用 float32（`model.py:183`），没有 AMX 的机器上 bf16 更慢 |
| 用系统自带的 Python 3.7/3.9 | ❌ Jeff 要求 `>=3.12`，脚本用 `uv python install 3.12` 单独装 |

### 6.4 确认 NPU 确实没被用到

CPU 模式下 Jeff 不该占用任何 NPU 算力，可以用这条确认（也用来排查是不是别的任务在抢卡）：

```bash
npu-smi info            # 看 AICore 利用率
watch -n 2 npu-smi info # 持续观察，推理时应始终为 0
```

如果推理时 AICore 有占用，说明 6.1 的隔离没生效，或者 `JEFF_DEVICE` 没设成 cpu。

### 6.5 性能参考

| 平台 | 单请求延迟预估 |
|---|---|
| 鲲鹏 920（aarch64） | 1–3s |
| Xeon（x86_64） | 0.5–1.5s |

比 Windows 桌面（0.75–0.97s）略慢是正常的 —— 服务器 CPU 单核频率通常低于桌面，且 Jeff 的串行推理吃不到多核红利。靠多实例把总吞吐拉起来。

---

## 七、性能预期与并发架构

| 项目 | 预期 |
|---|---|
| 单请求延迟 | 0.5–2s（取决于 CPU 主频与核数；Windows i7 实测 0.75–0.97s） |
| 单实例吞吐 | ≈ 1 QPS |
| 冷启动 | 权重加载 20–60s（1.6GB 从磁盘读入 + 建图） |

**单实例扛不住并发，用多实例 + 反向代理。** 建议实例数 ≈ `CPU 核心数 / 8`：

```bash
# 起 4 个实例
for p in 8765 8766 8767 8768; do
  JEFF_CHECKPOINT=/opt/jeff/checkpoints/jeff-0.8b JEFF_DEVICE=cpu \
  JEFF_HOST=127.0.0.1 PORT=$p JEFF_QUEUE_MS=2000 \
  nohup /opt/jeff/.venv/bin/jeff-serve > /var/log/jeff-$p.log 2>&1 &
done
```

Nginx 前置（自动重试 529）：

```nginx
upstream jeff {
    server 127.0.0.1:8765 max_fails=0;
    server 127.0.0.1:8766 max_fails=0;
    server 127.0.0.1:8767 max_fails=0;
    server 127.0.0.1:8768 max_fails=0;
}
server {
    listen 8760;
    location / {
        proxy_pass http://jeff;
        proxy_next_upstream error timeout http_529 non_idempotent;
        proxy_read_timeout 60s;
    }
}
```

> `proxy_next_upstream ... http_529` 是关键：把排队超时的请求自动转到下一个实例。

---

## 八、systemd 常驻

```bash
sudo cp deploy/jeff-cpu.service /etc/systemd/system/
sudo useradd -r -s /sbin/nologin jeff || true
sudo chown -R jeff:jeff /opt/jeff
sudo systemctl daemon-reload
sudo systemctl enable --now jeff-cpu
sudo journalctl -u jeff-cpu -f
```

单元文件内容见 `deploy/jeff-cpu.service`，里面已经设好 `JEFF_DEVICE=cpu`、`JEFF_QUEUE_MS=2000`、`Restart=always`。
如果要跑多实例，复制成 `jeff-cpu@.service` 用模板单元更省事。

---

## 九、验证部署

```bash
# 健康检查
curl -s http://127.0.0.1:8765/v1/models

# 真实推理（仓库里的样本）
curl -s http://127.0.0.1:8765/v1/systemone \
  -H 'content-type: application/json' \
  -d @test_request.json
```

期望结果与 Windows 一致：route → `shipping` 约 99.2%，noul 约 13.7%，score 约 1.336。

要用完整评测套件验证，把 `jeff-bench` 项目也拷过去（纯标准库，无额外依赖）：

```bash
python src/run_bench.py --url http://127.0.0.1:8765
```

---

## 十、后续：要不要上 NPU

| 方案 | 状态 |
|---|---|
| CPU（本笔记） | ✅ 现在就能跑，功能完整 |
| 改代码接 NPU | ⚠️ 卡在版本栈：TorchNPU 26.1.0 最高配套 PyTorch 2.12，Jeff 锁 2.14 |
| 拆分部署（vLLM-Ascend 跑底座 + 自写 readout 头） | ✅ 可行且更优：vLLM-Ascend 已官方适配 Qwen3.5-0.8B（Gated DeltaNet 算子现成），决策头只是 255×1024 矩阵乘，还能顺带绕开 529 并发瓶颈 |
| 等 TorchNPU 追到 2.14 | 华为迭代节奏看，2–3 个月内大概率有，届时改一行 `JEFF_DEVICE` 白名单即可 |

拆分方案值得提前准备的部分：从 checkpoint 导出 `readout.weight`、解析 255 个选项码 token、用 `chat_template.jinja` 复刻 Jeff 的 state-first prompt 布局 —— 这些都能先在 CPU 环境验证正确，再搬到 910B。

---

## 十一、踩坑速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `uv sync` 拖了几个 GB | Linux 默认拉 CUDA 版 torch | 先装 `+cpu` wheel，再 `--no-install-package torch --no-install-package torchvision` |
| 别的机器连不上 | `JEFF_HOST` 默认 127.0.0.1 | 显式设 `JEFF_HOST=0.0.0.0` |
| 大量 HTTP 529 | 单线程 + `JEFF_QUEUE_MS` 默认 0 | 设 `JEFF_QUEUE_MS=2000`，或 Nginx `proxy_next_upstream http_529` |
| 报 "no cuda device" 或张量设备错误 | torch_npu 劫持了 `cuda.is_available()` | 显式 `JEFF_DEVICE=cpu`，剥离 PYTHONPATH 里的 Ascend 条目（见 6.1） |
| 核很多但推理很慢 | `OMP_NUM_THREADS` 开太大 | 降到 8–16，改起多实例（见 6.2） |
| 怀疑 NPU 被占用 / 想确认模式 | — | `watch -n 2 npu-smi info`，CPU 模式 AICore 应始终为 0 |
| `torch` import 报 GLIBC 版本错 | 系统太老（glibc < 2.28） | 换 Ubuntu 20.04+ / openEuler 22.03+ |
| aarch64 上某些包装不上 | 缺 aarch64 wheel | 换 x86_64 机器，或从源码编译 |
| 权重下载卡住 | HuggingFace 直连不通 | `export HF_ENDPOINT=https://hf-mirror.com` |
