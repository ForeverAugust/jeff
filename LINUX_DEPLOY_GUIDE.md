# Jeff Linux 部署指导

从零把 Jeff 决策服务跑在 Linux 服务器上的完整手册。适用 Ubuntu 20.04+ / Debian 11+ / openEuler 22.03+ / CentOS 8+，架构 x86_64 或 aarch64。

配套文件：

| 文件 | 用途 |
|---|---|
| `deploy/install-cpu.sh` | CPU 模式一键部署（检测架构 → 装 CPU 版 torch → 下载权重 → 启动自检） |
| `deploy/jeff-cpu.service` | systemd 单元，CPU 模式常驻 |
| `deploy/ascend-check.sh` | 昇腾主机只读诊断 |
| `DEPLOY_NOTES_LINUX.md` | 昇腾 910B / 鲲鹏专项笔记（NPU 走 CPU 模式的坑与调优） |
| `DEPLOY_NOTES.md` | Windows 桌面部署笔记 |

---

## 一、30 秒选型

| 你的机器 | 走哪条路线 | 单请求延迟 | 说明 |
|---|---|---|---|
| NVIDIA GPU（显存 ≥ 8GB） | torch CUDA + `--extra cuda` | 0.05–0.2 s | **生产首选** |
| 无 GPU，x86_64 服务器 | torch CPU（`+cpu` wheel） | 0.5–1.5 s | 功能完整，靠多实例堆吞吐 |
| 无 GPU，aarch64（鲲鹏等） | torch CPU | 1–3 s | 同上，见昇腾笔记 §6.2 线程调优 |
| 无 GPU，但要高吞吐 | llama.cpp + GGUF | 0.2–0.5 s | 见 §9，CPU 上比 torch 快数倍 |
| 昇腾 910B / Atlas 800 A2 | torch CPU（**不支持 npu**） | 1–3 s | 见 `DEPLOY_NOTES_LINUX.md` |

> 为什么昇腾不支持 NPU：`src/jeff/models.py` 的 `device_from_environment()` 白名单只有 `cuda / mps / cpu`，写 `npu` 直接 `ValueError`。

---

## 二、系统要求

| 项目 | 要求 | 检查命令 |
|---|---|---|
| 架构 | x86_64 或 aarch64 | `uname -m` |
| glibc | ≥ 2.28（torch wheel 是 manylinux_2_28） | `ldd --version \| head -1` |
| Python | ≥ 3.12（文档统一用 3.12，wheel 最全） | `python3 --version` |
| 内存 | ≥ 8 GB（CPU 模式 float32 约 3.2 GB + 运行时） | `free -g` |
| 磁盘 | 权重 1.7 GB + venv 0.2–4 GB | `df -h` |
| 网络 | 能访问 PyPI 镜像；首次需下权重 | — |
| NVIDIA 驱动（仅 GPU 路线） | 支持 CUDA 13 的较新驱动 | `nvidia-smi` |

---

## 三、安装 uv 与代码

```bash
# 1. uv（项目要求 >= 0.12.19）
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version

# 2. 代码（国内直连 GitHub 不稳时走 gh-proxy）
git clone https://gh-proxy.com/https://github.com/ForeverAugust/jeff.git /opt/jeff
cd /opt/jeff

# 3. venv，固定 3.12
uv python install 3.12
uv venv --python 3.12
```

国内镜像（建议写进 shell 或 `/etc/profile.d/jeff.sh`）：

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 四、装 torch：两条路线的分水岭

### 4.1 NVIDIA GPU 路线

Linux 上 PyPI 的 `torch` 默认就是 CUDA 版，**直接 sync 即可**（但要接受 4GB+ 的 CUDA 依赖）：

```bash
uv sync --no-default-groups --extra cuda
```

`--extra cuda` 装 `flash-linear-attention` 和 `kernels`，是 Qwen3.5 混合线性注意力的快算子。没有它 transformers 会回退到慢得多的实现（能跑，但延迟可能翻好几倍）。这两个包首次运行会从 HuggingFace 拉内核，离线机器要提前准备缓存。

驱动验证：

```bash
nvidia-smi                      # 右上角 CUDA Version 需 >= 13
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.14.0+cu130 True
```

启动：`JEFF_DEVICE=cuda`（其实不设也行，检测到 CUDA 会自动用）。

### 4.2 CPU 路线（x86_64 / aarch64 / 昇腾）

**坑：Linux 直接 `uv sync` 会拖下来 4GB+ 的 CUDA 全家桶**（cuda-bindings、cuda-toolkit、nvidia-cudnn-cu13、nccl、triton…）。必须先手工装 `+cpu` wheel，再跳过：

```bash
ARCH=$(uname -m)                                  # x86_64 / aarch64
BASE="https://mirrors.aliyun.com/pytorch-wheels/cpu"
curl -LO "$BASE/torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_${ARCH}.whl"
curl -LO "$BASE/torchvision-0.29.0+cpu-cp312-cp312-manylinux_2_28_${ARCH}.whl"
uv pip install ./*.whl

UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  uv sync --no-default-groups --no-install-package torch --no-install-package torchvision
```

`2.14.0+cpu` 满足 `torch==2.14.0`（PEP 440 忽略 local version 段），不会触发版本冲突。

验证：

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.14.0+cpu False
```

> 清华源没有这两个 wheel，download.pytorch.org 国内常 403，阿里云 `pytorch-wheels/cpu` 两个架构都验证过可用。

---

## 五、下载权重

两个可选仓库，用途不同：

| 仓库 | 内容 | 何时用 |
|---|---|---|
| `jeff-legacy/Jeff-Qwen3.5-0.8B` | 通用决策基座（含 readout 头），1.7 GB | **只用基座做 zero-shot 决策，或自己做微调起点** |
| `mstrasser/jeff-base`（revision `v1.3`） | adapter-first 基座 | 要挂官方 LoRA 适配器（triage/spam/soc…） |

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 通用基座
uv run --no-default-groups hf download jeff-legacy/Jeff-Qwen3.5-0.8B \
  --local-dir checkpoints/jeff-0.8b

# 或：adapter-first 基座（配套 §11 的适配器）
uv run --no-default-groups hf download mstrasser/jeff-base --revision v1.3 \
  --local-dir jeff-base-v1.3
```

`hf` CLI 卡住时改用 curl 逐个下：

```bash
mkdir -p checkpoints/jeff-0.8b
for f in config.json decision_config.json chat_template.jinja tokenizer.json \
         tokenizer_config.json processor_config.json model.safetensors readout.safetensors; do
  curl -L -o "checkpoints/jeff-0.8b/$f" \
    "https://hf-mirror.com/jeff-legacy/Jeff-Qwen3.5-0.8B/resolve/main/$f"
done
```

核心文件核对（`model.safetensors` 必须 1,706,027,688 字节）：

| 文件 | 作用 |
|---|---|
| `model.safetensors` | 底座权重（Qwen3.5-0.8B） |
| `readout.safetensors` | 决策头，255×1024 线性层（约 522 KB） |
| `decision_config.json` | 架构声明、`max_options` 上限、出厂 temperature |
| `chat_template.jinja` | state-first 的 prompt 布局 |

---

## 六、启动服务

```bash
# CPU
JEFF_CHECKPOINT=checkpoints/jeff-0.8b JEFF_DEVICE=cpu \
JEFF_HOST=127.0.0.1 PORT=8765 JEFF_QUEUE_MS=2000 \
  uv run --no-default-groups jeff-serve

# GPU
JEFF_CHECKPOINT=jeff-base-v1.3 JEFF_DEVICE=cuda JEFF_ADAPTERS=adapters \
JEFF_HOST=127.0.0.1 PORT=8765 JEFF_QUEUE_MS=500 \
  uv run --no-default-groups --extra cuda --extra lora jeff-serve
```

冷启动 20–60 s（1.6 GB 权重读盘 + 建图）。看到 Uvicorn 的 `Application startup complete` 才算就绪。

### 环境变量全表

| 变量 | 默认值 | 说明 |
|---|---|---|
| `JEFF_CHECKPOINT` | `checkpoints/selected` | 权重目录，**必设** |
| `JEFF_DEVICE` | 自动（有 CUDA 就用） | `cuda` / `mps` / `cpu`，机器没有该设备会报错而非静默降级 |
| `JEFF_HOST` | `127.0.0.1` | 对外开放必须显式设 `0.0.0.0`（建议只在反代后面这么做） |
| `PORT` | `8000` | 监听端口 |
| `JEFF_QUEUE_MS` | `0` | 模型忙时请求排队时长；**默认 0 意味着并发直接 529**，务必设 500–2000 |
| `JEFF_MAX_TOKENS` | `8192` | 单问题最长输入，超出返回 422（不截断） |
| `JEFF_ADAPTERS` | 无 | LoRA 适配器目录；设了才能按 `model` 字段选适配器 |
| `JEFF_ADAPTER_MODE` | `shared` | `merged` 把单个适配器折进底座（速度=底座，但不能热换） |
| `JEFF_LORA_PRECISION` | `model` | `float32` 用参考实现精度 |
| `JEFF_BACKEND` | `pytorch` | `mlx` 走 Apple silicon |
| `JEFF_API_KEY` | 无 | 设了则要求 `Authorization: Bearer <key>` |
| `OMP_NUM_THREADS` | 系统默认 | CPU 模式关键，见 §10 |

---

## 七、systemd 常驻

```bash
sudo useradd -r -s /sbin/nologin jeff
sudo mkdir -p /var/log/jeff && sudo chown jeff:jeff /var/log/jeff
sudo cp deploy/jeff-cpu.service /etc/systemd/system/
sudo chown -R jeff:jeff /opt/jeff
sudo systemctl daemon-reload
sudo systemctl enable --now jeff-cpu
sudo journalctl -u jeff-cpu -f
```

`deploy/jeff-cpu.service` 已设好 `JEFF_DEVICE=cpu`、`JEFF_QUEUE_MS=2000`、`Restart=always`、日志落地与 systemd 加固项。

GPU 路线把单元文件里的 `Environment="JEFF_DEVICE=cpu"` 改成 `cuda`，`ExecStart` 换成含 `--extra cuda` 的 venv 里的 `jeff-serve`（venv 已装好依赖，`/opt/jeff/.venv/bin/jeff-serve` 即可）。

多实例（§8 必需）：

```bash
sudo cp deploy/jeff-cpu.service /etc/systemd/system/jeff-cpu@.service
# 把单元里的 Environment="PORT=8765" 改成 Environment="PORT=%i"
sudo systemctl enable --now jeff-cpu@8765 jeff-cpu@8766 jeff-cpu@8767 jeff-cpu@8768
```

---

## 八、并发：绕开 HTTP 529

**jeff-serve 一次只处理一个请求。** 模型忙时新请求等 `JEFF_QUEUE_MS` 后返回 **529 + `Retry-After: 1`**。这是设计，不是故障。

三件事一起做：

1. `JEFF_QUEUE_MS=2000`（给排队机会，超出才拒绝）
2. 多实例：`实例数 ≈ CPU 核心数 / 8`（GPU 上单卡 1–2 个实例）
3. 反代自动重试

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

> `proxy_next_upstream ... http_529` 是关键：把排队超时的请求转到下一个实例。

客户端侧兜底（Python 示例见 §12）。

| 部署形态 | 单实例吞吐 | 说明 |
|---|---|---|
| CPU，1 实例 | ≈ 1 QPS | 多实例线性提升 |
| GPU，1 实例 | 10–20 QPS | 单卡起 2 实例收益有限（模型已占满 SM） |

---

## 九、其他部署形态

### GGUF / llama.cpp（CPU 高吞吐）

上游提供了 GGUF 量化版（`mstrasser/jeff-base-gguf`，Q8_0 / Q4_K_M）与配套 LoRA GGUF，用 `llama-server` 加载，一条命令服务所有适配器（`-lora`）。CPU 服务器上吞吐显著高于 torch CPU 路线。

> 具体命令以 [jeffhub.ai/docs/llama-cpp](https://jeffhub.ai/docs/llama-cpp) 与所用 llama.cpp 版本为准；Q8_0 精度基本无损，Q4_K_M 需实测你的场景。

### 容器

上游未提供官方镜像。自建时的要点：基础镜像 `python:3.12-slim`（glibc 满足 2.28）、CPU 路线仍要先装 `+cpu` wheel、权重挂卷而非打进镜像、`JEFF_HOST=0.0.0.0`。

```dockerfile
FROM python:3.12-slim
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ uv
WORKDIR /opt/jeff
COPY . .
RUN uv venv --python 3.12 \
 && ARCH=$(uname -m) \
 && VENV=/opt/jeff/.venv/bin/uv \
 && $VENV pip install -i https://mirrors.aliyun.com/pypi/simple/ \
      "https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.14.0%2Bcpu-cp312-cp312-manylinux_2_28_${ARCH}.whl" \
 && UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
      uv sync --no-default-groups --no-install-package torch --no-install-package torchvision
ENV JEFF_CHECKPOINT=/opt/jeff/checkpoints/jeff-0.8b JEFF_DEVICE=cpu \
    JEFF_HOST=0.0.0.0 PORT=8765 JEFF_QUEUE_MS=2000
EXPOSE 8765
CMD ["/opt/jeff/.venv/bin/jeff-serve"]
```

---

## 十、CPU 模式性能调优

线程数不是越多越好。Jeff 是 0.8B 小模型的**短串行图**，线程开多了调度开销反超计算收益，实测 8–16 见顶：

| 逻辑核数 | `OMP_NUM_THREADS`（每实例） | 建议实例数 |
|---|---|---|
| ≥ 96 | 16 | 6–8 |
| 48–95 | 16 | 3–4 |
| 24–47 | 8 | 3–4 |
| 12–23 | 8 | 2 |
| < 12 | 4 | 1 |

```bash
export OMP_NUM_THREADS=16 OMP_PROC_BIND=false MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
```

其他注意：

| 事项 | 结论 |
|---|---|
| aarch64 上设 `TORCH_ARM_SVE=1` | ❌ 鲲鹏 920 是 ARMv8.2，不支持 SVE，无效 |
| CPU 下改 bfloat16 | ❌ Jeff 在 CPU 自动用 float32，无 AMX 的机器上 bf16 更慢 |
| 多实例 vs 多线程 | ✅ 多实例远优于单实例多线程（单实例串行是硬限制） |

---

## 十一、LoRA 适配器

官方适配器（triage / spam / guard / ground / soc / tools …）能把特定场景准确率从 20–50% 拉到 90%+。**注意授权：sanctions 和 soc 是 CC BY-NC 4.0，禁止商用**；aml 与 trading-desk 另有数据条款，见各自 jeffhub 页面。

```bash
# 装 lora 依赖
uv sync --no-default-groups --extra cuda --extra lora

# 下适配器（以 triage 为例）
for name in triage spam guard; do
  uv run --no-default-groups hf download mstrasser/jeff-adapter-$name --revision v1.3 \
    --local-dir adapters/$name
done

# 启动：按请求的 model 字段路由
JEFF_CHECKPOINT=jeff-base-v1.3 JEFF_ADAPTERS=adapters PORT=8765 \
  uv run --no-default-groups --extra cuda --extra lora jeff-serve
```

| 能力 | 做法 |
|---|---|
| 运行时增删适配器 | 往目录增删文件夹后 `POST /v1/adapters/reload`（无需重启） |
| 只要一个适配器、要底座速度 | `JEFF_ADAPTER_MODE=merged`（此时不再服务裸底座，也不支持热换） |
| 精度对齐参考实现 | `JEFF_LORA_PRECISION=float32` |

---

## 十二、调用 API

### 三种问题类型

| type | 字段 | 返回 |
|---|---|---|
| `choice` | `criteria`：选项名 → 描述（**dict，1–255 项**） | `probabilities` + `choice` + `confidence` |
| `noul` | `instructions` 一句陈述 | `noul`（0–1 的认同度） |
| `score` | `criteria`：有序档位列表（2–10 项） | `score`（连续值）+ `legend` + `probabilities` |

`state` 是情境描述；`questions` 可一次问多个（一次前向传播全部回答）；`orders: 2` 表示正反序各答一次取平均（双倍耗时，抗选项顺序偏置）；`images` 最多 4 张 base64 图。

```bash
curl -s http://127.0.0.1:8765/v1/systemone \
  -H 'content-type: application/json' \
  -d @test_request.json
```

```json
{
  "model": "jeff-latest",
  "state": "Refund request: the customer says the parcel arrived crushed...",
  "questions": {
    "route": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "payment issues", "shipping": "delivery damage",
                           "account": "login problems", "sales": "new purchases"}},
    "refund_eligible": {"type": "noul", "instructions": "The customer is eligible for a full refund."},
    "priority": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}
  }
}
```

Python 客户端（自带 529 退避）：

```python
from jeff import Client
from jeff.client import choice_question, yes_no_question

jeff = Client("http://127.0.0.1:8765", model="triage")
answers = jeff.ask("The parcel arrived crushed and I want my money back.", {
    "route": choice_question("Which team?", {"billing": "payment", "shipping": "delivery damage"}),
    "refund": yes_no_question("The customer is eligible for a full refund."),
})
```

> 两条铁律：**不要用裸数字当选项名**；把固定不变的内容放进 `state`、变化的放进问题（prompt 是 state-first 布局）。详见仓库 `docs/v1.3-request-format.md`。

### 状态码

| 码 | 含义 | 处理 |
|---|---|---|
| 200 | 正常 | — |
| 401 | `JEFF_API_KEY` 不匹配 | 带 `Authorization: Bearer <key>` |
| 422 | 请求不合规范（选项超限、输入超 `JEFF_MAX_TOKENS`、模型名未知） | 看响应体 |
| 503 | 模型还没加载完 | 等冷启动结束 |
| 529 | 模型忙且排队超时 | 退避重试，或交给 Nginx 转下一个实例 |

---

## 十三、微调（`jeff-train`）

**必须有 GPU**，CPU 上不现实（`--cpu-threads` 只是数据加载线程）。

```bash
uv sync --no-default-groups --extra cuda --extra lora      # LoRA 需要 lora 组

# 全权重 SFT
uv run jeff-train --run runs/triage-01 --output artifacts/triage \
  --initial-checkpoint jeff-base-v1.3 --train data/train.jsonl \
  --base-model jeff-base-v1.3 --epochs 1 --lr 2e-6

# LoRA（从冻结的初始检查点训低秩适配，显存友好）
uv run jeff-train --run runs/triage-lora --output artifacts/triage-lora \
  --initial-checkpoint jeff-base-v1.3 --train data/train.jsonl \
  --lora-rank 16 --lora-alpha 32
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--run` / `--output` | 必填 | 运行名 / 产物目录 |
| `--base-model` | `Qwen/Qwen3.8-27B` | 0.8B 场景指向本地基座目录 |
| `--initial-checkpoint` | 无 | 起点权重；LoRA 模式必填 |
| `--lora-rank` | 无 | 给此值即进入 LoRA 模式 |
| `--epochs` / `--lr` | 1 / 2e-6 | — |
| `--batch-size` / `--effective-batch-size` | 32 / 256 | 按显存调 |
| `--max-length` / `--token-budget` | 8192 | 与训练长度一致 |
| `--patience` | 无 | 验证损失不再下降就早停 |
| `--resume` | 无 | 断点续训 |

训练完用 `jeff-kit` 校验数据（切分/泄漏/短路检查），用 `jeff-evaluate` 评估：

```bash
uv run jeff-kit check --train data/train.jsonl --dev data/dev.jsonl
uv run jeff-evaluate --checkpoint artifacts/triage --data data/dev.jsonl
```

---

## 十四、验证部署

```bash
# 就绪检查
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/v1/models

# 真实推理（仓库自带样本）
curl -s http://127.0.0.1:8765/v1/systemone -H 'content-type: application/json' -d @test_request.json
```

期望：route → `shipping` 约 99.25%，`refund_eligible` noul 约 0.137，priority `score` 约 1.336（与 Windows 部署一致）。

更全面的验证用 `jeff-bench`（`D:\code2026\jeff-bench`，纯标准库无额外依赖，46 个用例）：

```bash
python src/run_bench.py --url http://127.0.0.1:8765
```

延迟基准用仓库自带工具：

```bash
uv run jeff-latency --url http://127.0.0.1:8765
```

---

## 十五、安全加固

| 项目 | 做法 |
|---|---|
| 访问控制 | 设 `JEFF_API_KEY`，客户端带 `Authorization: Bearer <key>`（用 hmac 常量时间比较） |
| 监听面 | 默认 `127.0.0.1`；只在有反代/防火墙时才 `0.0.0.0` |
| 运行用户 | 用 `jeff` 系统用户（`/sbin/nologin`），目录属主归它 |
| 防火墙 | 只放行反代端口：`sudo ufw allow 8760/tcp` |
| systemd 加固 | `NoNewPrivileges`、`PrivateTmp`、`ProtectSystem=full`（单元文件已带） |
| 授权合规 | soc / sanctions 适配器 CC BY-NC 4.0，商用前确认授权 |

---

## 十六、排障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `uv sync` 拖了几个 GB | Linux 默认拉 CUDA 版 torch | 先装 `+cpu` wheel，再 `--no-install-package torch torchvision` |
| 别的机器连不上 | `JEFF_HOST` 默认 `127.0.0.1` | 显式 `JEFF_HOST=0.0.0.0` |
| 大量 HTTP 529 | 单线程 + `JEFF_QUEUE_MS` 默认 0 | 设 `JEFF_QUEUE_MS`；多实例 + Nginx `proxy_next_upstream http_529` |
| `ValueError: JEFF_DEVICE='npu'` | 白名单只有 cuda/mps/cpu | 昇腾上写 `cpu` |
| 报 "no cuda device" 或张量设备错 | `torch_npu` 劫持了 `cuda.is_available()` | 显式 `JEFF_DEVICE=cpu`，剥离 `PYTHONPATH` 里的 Ascend 条目（见昇腾笔记 §6.1） |
| 核很多但推理慢 | `OMP_NUM_THREADS` 开太大 | 降到 8–16，改起多实例 |
| `torch` import 报 GLIBC 版本错 | glibc < 2.28 | 换 Ubuntu 20.04+ / openEuler 22.03+ |
| 权重下载卡住 | HuggingFace 直连不通 | `export HF_ENDPOINT=https://hf-mirror.com`，或用 curl 逐个下 |
| GPU 上很慢 | 没装 `--extra cuda`，走了慢回退 | 装上 `flash-linear-attention` + `kernels` |
| kernels 首次运行失败 | 需联网从 HF 拉内核 | 提前预热缓存，或离线环境预置 |
| aarch64 某些包装不上 | 缺 aarch64 wheel | 换 x86_64 或源码编译 |
| 422 "Unknown model" | 模型名不在别名/适配器列表 | 用 `jeff-latest`，或 `/v1/models` 查真实名 |
| 422 输入过长 | 超 `JEFF_MAX_TOKENS` | 调大（>8192 的精度未测），或缩短输入 |
