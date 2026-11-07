# 安装与配置

## 运行环境

安装和启动脚本只支持 **Windows x64 + NVIDIA CUDA**。所有运行环境都装在项目目录里（`.venv/` 和 `.runtime/`），不修改系统里的 Python。

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.15（由安装程序下载） |
| PyTorch | 2.8.0+cu128（CUDA runtime 12.8，不需要另装 CUDA Toolkit） |
| Gradio | 5.45.0 |
| 语音识别 | SenseVoiceSmall Q8，CPU 运行，需要[单独导入](#导入-asr-程序和模型) |

请用安装脚本安装，不要用 `pip install` 或 `uv sync` 代替：训练依赖、FFmpeg 和加速组件都由脚本另外安装。

## 安装

在项目根目录双击 **`1- install.bat`**。安装程序依次完成以下步骤：

1. 检查 NVIDIA 驱动（`nvidia-smi`）；
2. 下载项目自带的 uv 和 Python；
3. 按锁文件安装依赖，并验证 PyTorch、CUDA 和 NumPy；
4. 准备文本规范化环境：独立的 NeMo / Conda 环境，失败时自动使用简化方案，不影响安装；
5. 安装 LoRA 训练所需的附加依赖；
6. 安装 Triton 和 FlashAttention 加速组件。FlashAttention 安装失败只会给出警告，基础推理不受影响。

安装阶段不下载模型。重新运行 `1- install.bat` 可以修复不完整的环境。

## 启动

双击 **`2- run.bat`**。启动时会：

- 检查运行环境（CUDA、NumPy、FFmpeg）；
- 检查 `models/IndexTTS-2.5/`，缺少的模型和辅助资源会从 Hugging Face 自动下载；
- 在 <http://127.0.0.1:7860> 打开界面（地址以启动窗口显示的为准）。

语料识别用的 ASR 程序和模型**不会**自动下载，请按下一节导入。

## 导入 ASR 程序和模型

“数据准备”页的批量识别，以及参考音频的自动转写，都使用 SenseVoiceSmall Q8 在 CPU 上运行。不需要识别功能的话可以跳过本节。

1. 下载 [FunASR Windows CPU 便携运行时](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) 和 [SenseVoiceSmall Q8 GGUF 模型](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)，整理成下面的结构：

   ```text
   funasr-llamacpp-windows-x64/
   ├─ llama-funasr-sensevoice.exe
   └─ models/
      └─ sensevoice-small-q8.gguf
   ```

2. 在项目根目录运行（换成你自己的路径）：

   ```powershell
   .\.venv\Scripts\python.exe tools\import_funasr_runtime.py `
     --source 'D:\Tools\funasr-llamacpp-windows-x64'
   ```

导入程序会校验模型的 SHA-256，然后复制文件：识别程序放在 `.runtime/funasr/`，模型放在 `models/asr/`。重复导入相同的文件不会出问题；如果目标位置已有不同的文件，导入会停止，确认要替换时再加 `--replace`。用 `--root <目录>` 可以导入到另一个部署目录。

## 本地设置

把 `local-settings.example.bat` 复制一份，重命名为 `local-settings.bat`，即可修改监听地址、端口和代理。安装、启动、更新三个脚本都会读取这个文件，它不会被 Git 跟踪，也不会被更新覆盖。

```bat
set "INDEXTTS_HOST=127.0.0.1"
set "INDEXTTS_PORT=7860"
rem 需要代理时取消注释，并改成自己的地址
rem set "HTTP_PROXY=http://127.0.0.1:7890"
rem set "HTTPS_PROXY=http://127.0.0.1:7890"
```

**局域网访问**：把 `INDEXTTS_HOST` 改成这台电脑的局域网 IP。注意，界面里填写的语料、模型、导出路径都是**运行 GUI 的那台电脑**上的路径；“浏览”“另存为”等系统窗口也会在那台电脑上弹出，远程操作时请直接输入路径。

**界面语言**：在界面顶部的“界面语言 / UI Language”中切换，重启后生效。界面语言和合成运行选项保存在 `.runtime/gui_settings.json`。

## 可选加速组件

| 组件 | 作用 | 条件 |
| --- | --- | --- |
| BF16 | 降低显存占用、提高速度 | 显卡支持即可 |
| Triton / torch.compile | 编译 S2Mel 模块，加快推理 | 需要 Triton；首次编译较慢 |
| FlashAttention | 加速 GPT 注意力计算 | 需要 FlashAttention wheel、Triton 和 BF16 |

安装程序会自动安装 Triton 和 FlashAttention。如果界面提示组件不可用，重新运行 `1- install.bat` 即可。如果你有与当前环境完全匹配的 `flash_attn` wheel（Python 3.11、PyTorch 2.8、CUDA 12.8），把它放进 `accelerator_wheels/` 再重新安装，安装程序会优先使用它。

加速收益取决于具体任务。遇到问题时，可以先关闭这些选项，用基础路径确认能正常合成。

## 数据存放位置

以下目录都在项目根目录下，除 `models/asr/README.md` 外都不纳入 Git，更新时也会保留。

| 路径 | 内容 |
| --- | --- |
| `models/IndexTTS-2.5/` | TTS 模型权重与配置 |
| `models/asr/` | ASR 模型 |
| `voices/`、`emotion_descriptions/` | 声音库、情绪描述库 |
| `training/projects/`、`training/datasets/` | 训练项目设置、训练清单 |
| `training/outputs/` | LoRA 检查点和评估音频 |
| `outputs/` | 合成的音频、导出的检查点 |
| `.venv/`、`.runtime/` | 运行环境、缓存、设置、日志 |
| 语料目录里的 `.proofread/` | 校对进度和文本备份 |
| `local-settings.bat` | 本机的监听地址和代理设置 |
