# IndexTTS 2.5 Easy GUI

面向 Windows 的本地语音工作台，将 **语音合成、语料识别与校对、GPT LoRA 微调、模型导出** 放在同一个项目中。默认提供简体中文界面，可切换 English。

A Windows GUI for IndexTTS 2.5 with voice cloning, emotion control, multi-speaker synthesis, SenseVoice transcription, audio proofreading, and experimental GPT LoRA fine-tuning.

日常训练流程：**准备音频切片 → ASR 补充文本 → 听音校对 → 更新训练清单 → LoRA 微调 → 试听与导出**。

## 功能概览

| 模块 | 当前功能 |
| --- | --- |
| 参考音频库 | 上传或录音、试听、保存音频与可选同名 TXT，供推理和对话复用 |
| 语音合成 | 参考音频克隆、LoRA 加载、长文本分段、语速与采样控制；界面提供中文、英文、日文、西班牙文和阿拉伯文语言选项 |
| 情绪与对话 | 参考音频情绪、独立情绪音频、八维情绪向量、文字情绪描述；按角色组织多轮对话并拼接 WAV |
| 语料识别与校对 | SenseVoiceSmall Q8 CPU 批量补缺、单条重识别、递归目录扫描、倍速试听、文本编辑和校对进度 |
| LoRA 微调 | 项目管理、训练清单检查、Steps / Epochs、参数预设、检查点、续训、评估音频和 TensorBoard |
| 导出与运行状态 | CPU 合并完整 GPT `.pth`、运行日志、显存状态，以及可用时的 BF16、Triton / torch.compile、FlashAttention 推理选项 |

## 环境与安装

当前安装和启动脚本面向 **Windows x64 + NVIDIA CUDA**。项目使用独立的 `.venv` 和 `.runtime`，由安装器准备 Python，无需先修改系统 Python。

| 组件 | 当前脚本固定版本或行为 |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.8.0+cu128，CUDA runtime 12.8 |
| GUI | Gradio 5.45.0 |
| ASR | 独立 SenseVoiceSmall Q8 原生程序，CPU 推理，需单独导入 |
| FFmpeg | 使用项目内程序；启动时从已安装的 imageio-ffmpeg 准备 |

运行需要兼容的 NVIDIA 驱动；安装器会检查 `nvidia-smi`，启动器会验证 PyTorch 和 CUDA。CPU ASR 不代表整个 TTS / 训练项目可以只用 CPU。显存和内存占用随音频长度、推理设置和训练参数变化；界面中的显存预设不是最低硬件要求或显存适配保证。

### 1. 安装项目环境

下载或克隆源码，在项目根目录双击 **`1- install.bat`**，或在 PowerShell 中运行：

```powershell
& '.\1- install.bat'
```

安装器同步固定环境、训练依赖和推理加速组件。Windows 文本规范化使用独立的 NeMo / Conda 环境，与主 `.venv` 隔离；无法准备完整环境时使用代码提供的回退路径。FlashAttention 安装失败不会阻止基础 CUDA 推理环境安装完成。

初次安装和模型下载需要网络。不要仅用一次普通的 `pip install` 或 `uv sync` 代替完整安装流程：部分训练、FFmpeg 和加速依赖由安装脚本另行安装。

### 2. 启动 GUI

双击 **`2- run.bat`**，或运行：

```powershell
& '.\2- run.bat'
```

启动时检查运行环境及 IndexTTS 模型，缺失的 TTS 模型与辅助资源会自动下载。源码版本默认访问地址为 [http://127.0.0.1:7860](http://127.0.0.1:7860)；已有部署可能保留不同的本地配置，以启动窗口显示的地址为准。

**SenseVoice ASR 程序和模型不随 TTS 启动自动下载**，需要按下一节导入。

### 3. 导入 ASR 程序与模型

准备 [FunASR Windows CPU 便携运行时](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) 与 [SenseVoiceSmall Q8 GGUF 模型](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)。待导入目录应包含：

```text
funasr-llamacpp-windows-x64/
├─ llama-funasr-sensevoice.exe
└─ models/
   └─ sensevoice-small-q8.gguf
```

在 Easy GUI 根目录运行，将示例路径替换为自己的便携包目录：

```powershell
.\.venv\Scripts\python.exe tools\import_funasr_runtime.py `
  --source 'D:\Tools\funasr-llamacpp-windows-x64'
```

导入器会校验模型 SHA-256、核对复制结果，并生成 `.runtime/funasr/import.json`。重复导入相同文件可直接复用；目标已有不同文件时默认停止。`--root` 可指定另一部署目录，`--replace` 用于明确替换不同的已安装运行时。

导入后，程序位于 `.runtime/funasr/`，模型位于 `models/asr/`。模型目录通过 [README](models/asr/README.md) 保留在 Git 中，**GGUF 权重不纳入版本管理**。

## 语料准备、识别与校对

工作台处理已经切片的音频及同名 TXT，例如：

```text
my_voice/
├─ 001.wav
├─ 001.txt
├─ 002.wav             # 缺少 TXT 时可批量识别
└─ session_02/
   ├─ 001.wav
   └─ 001.txt
```

支持递归扫描 WAV、FLAC、MP3、M4A、OGG、AAC，不同子目录的同名文件按相对路径显示。正文应对应音频里实际说出的内容；例如日文音频应使用日文原台词，而不是翻译字幕。

1. 在“数据集准备”页创建或选择项目，填写语料目录和训练语言，点击“保存项目”。
2. 在内嵌校对工作台点击“识别缺失文本”。程序处理整个所选目录及子目录中的缺失、空白 TXT，已有正文跳过；列表搜索和筛选不改变批量识别范围。
3. 逐条试听、修改正文。点击“保存”只保存文本；“保存并下一条”还会标记当前条目已校对。
4. 需要重识别时，点击“重新识别当前条目”。结果先进入编辑框，由用户检查并保存，不自动覆盖 TXT。
5. 点击工作台的“更新训练清单”，然后进入 LoRA 训练。

批量识别可停止，已完成的 TXT 保留；再次运行会跳过已有正文。任务遇到错误停止，并显示诊断信息。所有 GUI 转写入口共用 CPU 后端和串行调度，包括参考音频库、推理参考音频和训练评估参考音频。

### 文本与进度保护

- TXT 保存为 UTF-8，首次改写已有文本时备份到语料目录的 `.proofread/backups/`，后续保存不覆盖首次备份。
- `.proofread/state.json` 保存校对进度，兼容原独立工具的第一层 WAV 记录。“已校对”仅记录进度，不限制训练样本是否被收录。
- 外部程序或其他页面修改 TXT 后，旧页面保存会报告冲突，保留编辑内容。文本改变也会使旧校对标记失效。
- 切换条目或目录时处理未保存修改；支持文件搜索、状态筛选、播放速度调整和进度拖动。
- 原音频保持不变；识别使用独立临时副本，转换为 16 kHz 单声道 PCM16，并规避原生程序的中文路径问题。

快捷键：`Ctrl+S` 保存，`Ctrl+Enter` 保存并下一条，`Alt+Space` 播放 / 暂停。

### 保存 TXT 后为什么还要更新清单？

训练读取的是 `training/datasets/<project>/manifest.jsonl`，其中的文本是生成清单时的快照。识别或校对保存不会自动重建清单。

更新清单会读取已保存 TXT，沿用音频时长筛选规则，写入 `text`、`reference_text`、语言和统计信息，不再次调用 ASR。**扫描到的文件数不一定等于最终训练样本数**，应检查有效样本统计。

GUI 管理的项目在启动训练前会检查清单是否过期，包括外部修改文本、更换语料目录等情况。正在训练或对应目录仍在批量识别时不能更新清单；训练期间仍可校对，修改供下次更新后使用。手动指定的外部清单保持原有行为，不自动绑定这套同步检查。

完整流程和故障处理见 [听音校对与 FunASR 转写](docs/proofreading-asr.md)。

## LoRA 微调与试听

当前训练器是 **实验性的 GPT acoustic-only LoRA**：通过 PEFT 训练自回归 GPT 的 LoRA 参数，其余 IndexTTS 模块保持冻结。训练使用原生声学 mel-token 重建目标，文本作为 TTS 条件输入；当前训练路径不启用额外文本、CTC 或 Whisper ASR 损失。

1. 在“LoRA 训练”页选择已更新清单的项目，设置独立的 Adapter 名称。
2. 按需要使用 AutoTune，再检查训练长度、Rank、Alpha、Dropout、学习率、梯度累积和保存间隔。AutoTune 是根据语料统计生成的参数起点，不会实测当前 GPU 的可用显存。
3. 可配置检查点评估文本和参考音频，开启评估音频生成；使用 TensorBoard 查看训练记录。
4. 先完成短程训练与试听，再安排长程训练。比较不同实验时使用不同输出名称，并用相同参考音频、相同未见过的文本试听，不能只凭训练 loss 选择权重。

输出保存在：

```text
training/outputs/<adapter>/
├─ checkpoint-step-000100/   # 示例：保存间隔由训练设置决定
├─ ...
└─ final_adapter/
```

GUI 支持选择检查点续训，也可在推理页选择已保存的检查点或 `final_adapter`。LoRA 是适配器，不是完整的基础模型，推理仍需要原有 IndexTTS 模型和参考音频。

## 语音合成与多角色对话

在参考音频库保存音频与可选文本后，可在推理页复用，也可直接上传参考音频。输入目标文本，选择语言及可选 LoRA 后生成语音。

- **长文本**：可按段落、句子、行或角色轮次分段，并设置段间静音。
- **情绪控制**：使用声音参考的自然情绪、单独的情绪参考音频、八维情绪向量，或 QwenEmotion 文字描述；文字描述可保存为预设。
- **Dialogue Builder**：为各轮选择声音、文本和情绪，支持添加、复制、排序、清空及删除轮次，最终按配置的间隔拼接 WAV。
- **运行选项**：BF16、GPT latent、torch.compile / Triton 和 FlashAttention 按当前环境能力启用。首次编译可能较慢；FlashAttention 路径在本 GUI 中需要 BF16、Triton 和对应 wheel，速度收益取决于实际任务。
- **状态查看**：界面提供生成进度、模型与显存状态、嵌入式控制台，以及输出保存选项。

参考文本描述的是参考音频中的原话，与要合成的目标文本不同。相关兼容字段仍保留，当前 acoustic-only 训练不学习额外的参考文本条件信号。

## 导出完整 GPT checkpoint

`tools/merge_lora_to_pth.py` 在 CPU 上将 LoRA 合入训练时的原始 FP32 `gpt.pth`，输出完整 GPT checkpoint，供兼容的 ComfyUI IndexTTS-2.5 节点加载。

```powershell
.\.venv\Scripts\python.exe -B tools\merge_lora_to_pth.py `
  --base models/IndexTTS-2.5/gpt.pth `
  --config models/IndexTTS-2.5/config.yaml `
  --adapter training/outputs/my_voice/final_adapter `
  --output outputs/merged/my_voice.pth
```

导出不覆盖基础模型、适配器或已有输出；会检查权重结构、精度、有限值、合并前后误差及保存回读，并写出 `.merge.json` 报告。可用 `--expected-base` 指定部署基础模型，核对其 SHA-256 与训练底模是否一致。

合并文件仅替代 GPT 权重，仍需要其他 IndexTTS 模型组件。对于提供 `GPT Checkpoint Selector` 的 ComfyUI 插件，可将导出文件放入 `models/IndexTTS-2.5/gpt_checkpoints/` 后选择加载；保留原始 `gpt.pth`。

## 本地配置与目录

源码启动脚本会读取可选的 `local-settings.bat`。将 `local-settings.example.bat` 复制并重命名后，可设置：

```bat
set "INDEXTTS_HOST=127.0.0.1"
set "INDEXTTS_PORT=7860"
rem 按需启用代理，并填写自己的地址。
rem set "HTTP_PROXY=http://127.0.0.1:7890"
rem set "HTTPS_PROXY=http://127.0.0.1:7890"
```

需要局域网访问时，将监听地址改为服务器的局域网 IP。远程浏览器中的语料路径仍指向服务器文件系统；GUI 的系统目录选择窗口也出现在服务器电脑上，可直接填写服务器路径完成选择。

界面语言和推理运行设置保存在 `.runtime/gui_settings.json`。切换 GUI 语言后按界面提示重启。

| 路径 | 内容 | Git 管理 |
| --- | --- | --- |
| `easy_gui.py`、`indextts_training/`、`tools/` | GUI、训练、ASR 和校对实现 | 是 |
| `models/IndexTTS-2.5/` | TTS 基础权重与配置 | 否 |
| `models/asr/` | Q8 ASR 模型 | 仅保留 README，权重忽略 |
| `voices/`、`emotion_descriptions/` | 本地声音库和情绪描述 | 否 |
| `training/projects/`、`training/datasets/` | 项目设置和训练清单 | 否 |
| `training/outputs/`、`outputs/` | LoRA、检查点、评估及合成输出 | 否 |
| `.runtime/`、`.venv/` | 运行时、环境、缓存、设置与日志 | 否 |
| 语料目录中的 `.proofread/` | 文本备份及校对进度 | 否 |
| `local-settings.bat` | 本机监听和代理配置 | 否 |

## 验证与常见问题

以下针对性检查使用临时样本或模拟进程，不启动真实训练；前端流程测试需要 Node.js：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_funasr_asr.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_proofread.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_dataset_proofread.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_easy_gui_i18n.py
node tests/test_proofread_frontend.cjs
```

- **ASR 提示缺少程序或模型**：执行运行时导入命令，检查 `.runtime/funasr/import.json` 和 `models/asr/`。GUI 不会静默回退 Whisper。
- **识别失败、空结果或超时**：查看错误中给出的 `.runtime/asr/logs/` 目录。单条任务超时为 180 秒；修复后可重新补缺。
- **清单待更新或无法更新**：先保存项目目录与训练语言、保存编辑框正文，等待训练和批量识别结束，再更新清单。
- **安装后仍存在 Whisper 依赖**：当前 GUI 转写已切换到 SenseVoice，但安装器及底层 tokenizer 等代码仍保留相关依赖，不应据此判断 GUI 还在使用旧转写模型。
- **CUDA / NumPy 验证失败**：查看启动窗口的具体版本或导入错误，使用项目安装脚本修复固定环境。缺少加速组件时，先使用基础推理路径验证。

## 上游项目与许可

本分支基于 [Mixomo / IndexTTS-2.5-Easy-GUI](https://github.com/Mixomo/IndexTTS-2.5-Easy-GUI)，保留上游历史、作者信息与许可证。相关项目包括：

- [IndexTTS](https://github.com/index-tts/index-tts)：基础语音模型与推理实现。
- [FunASR](https://github.com/modelscope/FunASR)、[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)：语音识别及 GGUF 运行时。
- [Triton for Windows](https://github.com/triton-lang/triton-windows)、[原 Windows 移植项目](https://github.com/woct0rdho/triton-windows)、[FlashAttention](https://github.com/Dao-AILab/flash-attention)、[Windows wheels](https://github.com/kingbri1/flash-attention)：可选推理加速组件。
- [Voice Clone Studio](https://github.com/FranckyB/Voice-Clone-Studio)：上游 GUI 工作流设计参考。

使用与分发遵循本仓库的 [LICENSE](LICENSE)、[中文许可证](LICENSE_ZH.txt)，以及各第三方程序和模型的上游许可。本分支的修改不代表原模型权利人的认可、担保或保证。
