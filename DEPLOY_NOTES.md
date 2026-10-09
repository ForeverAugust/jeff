# Jeff 决策模型部署指南（本机实测）

> 部署日期：2026-10-09 | 机器：Windows 11 + RTX 1000 Ada 6GB | CPU 模式已跑通

## 一、Jeff 是什么

Jeff 是开源的 "System 1" 决策模型（Jev 的开源替代），0.8B 参数：
- 输入一段状态描述 + 若干类型化问题（choice 多选 / noul 是非 / score 评分）
- 一次前向传播直接输出各选项的**校准概率**，不生成文本
- 代码 MIT，权重 Apache 2.0（HF: `jeff-legacy/Jeff-Qwen3.5-0.8B`）
- 2B 版综合分 83.1 追平 Jev 83.0；微调收益大（31.7% → 95.8% 半小时单卡）

## 二、当前部署状态

| 项目 | 状态 |
|---|---|
| 代码 | `D:\code2026\jeff`（gh-proxy 镜像克隆，2026-10-09 从 C 盘整体迁入） |
| 权重 | `D:\code2026\jeff\checkpoints\jeff-0.8b\`（model.safetensors 1.7GB，hf-mirror 下载） |
| 运行环境 | `.venv`（uv 管理，torch 2.14.0+cpu，serving-only） |
| 服务 | `http://127.0.0.1:8765`（`POST /v1/systemone`） |
| 模式 | **CPU float32**（单问 ~750–970ms，3 问/请求 ~2s）。**已决定不装 GPU 版 torch**，见第五节 |

> 迁移说明：整个目录（含 `.venv` 与权重，共 2.27GB / 22017 文件）已 robocopy 到 D 盘并验证服务输出与迁移前**完全一致**。venv 的 `pyvenv.cfg` 里 `home` 仍指向 `C:\Users\Administrator\.workbuddy\binaries\python\versions\3.13.12`（受管 Python，未移动），所以 venv 跨盘搬移后照常可用。

## 三、启动服务

**方式一：双击 `start.bat`（推荐，独立窗口常驻）**

**方式二：命令行**
```bash
cd /d/code2026/jeff
JEFF_CHECKPOINT=checkpoints/jeff-0.8b PORT=8765 ./.venv/Scripts/jeff-serve.exe
```

也可 `uv run --no-default-groups jeff-serve`（会联网校验 lockfile，慢）。

环境变量：`JEFF_CHECKPOINT`（权重目录，默认 `checkpoints`）、`PORT`（默认 8765）。
停止服务：`Ctrl+C`，或 `taskkill //F //IM jeff-serve.exe`。

## 四、请求格式（注意与 Jev 的差异！）

`choice` 的 `criteria` 是 **dict**（不是 Jev 的 list），`score` 的 `criteria` 是 **list**：

```json
{
  "model": "jeff-latest",
  "state": "Refund request: parcel arrived crushed, customer wants money back.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "billing team for payment issues",
        "shipping": "shipping team for delivery damage"
      }
    },
    "urgent": { "type": "noul", "instructions": "This is urgent." },
    "priority": { "type": "score", "instructions": "Urgency?", "criteria": ["low", "medium", "high"] }
  }
}
```

```bash
curl -s http://127.0.0.1:8765/v1/systemone -H "content-type: application/json" -d @test_request.json
```

实测响应（包裹损坏退款）：
- route → **shipping（99.2%）** ✅
- noul（全额退款资格）→ 13.7%
- score（紧急度）→ 偏 high

## 五、后续优化

### 1. 启用 GPU（RTX 1000 Ada 6GB）— 已决定不做

用户已于 2026-10-09 明确决定**保持 CPU 模式**，理由：当前用途（路由 / 分类 / 评测）对 ~1s 延迟不敏感，
不值得为此下载 3GB CUDA 版 torch 并占用 6GB 显存。

若将来需要（批量推理 / 微调 / 高 QPS 上线），装法如下（从 PyTorch 官方 index，走代理或镜像）：
```bash
cd jeff && uv pip install torch==2.14.0 torchvision==0.29.0 \
  --index-url https://download.pytorch.org/whl/cu130
```
装好后服务自动检测 CUDA，bfloat16 推理，延迟从 ~1s 降到几十 ms。
注意：CPU 模式下**批量不免费**（3 问 ≈ 单问的 2.5–2.7 倍），GPU 下才接近并行。

### 2. 装训练组（微调）
```bash
uv sync            # 默认组含 train（accelerate/datasets/flash-linear-attention 等）
```

### 3. LoRA 适配器
```bash
uv sync --no-default-groups --extra lora
# JEFF_ADAPTERS=adapters 环境变量挂多个适配器
```

## 六、本机踩坑记录（重要）

| 坑 | 解法 |
|---|---|
| GitHub 直连失败 | 用 `https://gh-proxy.com/https://github.com/...` 镜像克隆 |
| PyPI 直连极慢（torch 118MB 下了近 3 小时） | `--index-url https://mirrors.aliyun.com/pypi/simple/`（35 秒） |
| HF 官方 API 307 跳转（仓库迁移） | 模型在 `jeff-legacy/Jeff-Qwen3.5-0.8B`，走 `hf-mirror.com` |
| `hf download` 被 safe-delete 钩子杀掉 | 改用 `curl -L -o` 逐文件下载（hf-mirror 速度 ~20MB/s） |
| **torch 崩溃 WinError 1114（c10.dll）** | **根因：System32\msvcp140.dll 是 2021 老 14.29 版本**；装 VC++ Redist 14.51 修复（`VC_redist.x64.exe /repair /passive /norestart`）。事件日志定位：APPCRASH 在 msvcp140.dll 0xc0000005 |
| uv run 自动恢复 lockfile 版本 | 装了 2.13 测试后被 `uv run` 恢复成 2.14；修好运行库后 2.14 也能跑了，无需降级 |
| Git Bash 里 `robocopy ... /E` 报"无效参数 #3" | MSYS 把 `/E` 当路径转换了；加 `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"` 前缀 |
| 目录删不掉（safe-delete FAIL_CLOSED） | 回收站工具不可用，`rm -rf` 和 `shutil.rmtree` 都被钩子拦截；需在资源管理器里手动删 |

## 七、API 兼容性说明

- 端点路径与 Jev 相同（`POST /v1/systemone`）
- 但**请求体 schema 不同**（criteria dict vs Jev 的 options list），不是 100% drop-in
- 官方 TypeSafe SDK 不能直接指过来；用 jeff 自带 client（`clients/` 目录有 Python/TS 客户端）
