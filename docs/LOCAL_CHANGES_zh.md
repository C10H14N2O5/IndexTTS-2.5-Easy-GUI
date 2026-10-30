# 此分支的本地增强

本分支从上游 `Mixomo/IndexTTS-2.5-Easy-GUI` 的 `7cd880c` 提取可复用修改，保留原始历史、许可证与作者信息。运行环境和用户数据独立于源码仓库。

## Windows 环境与本地配置

- 安装器使用 uv 0.13.6，以支持项目固定的 Python 3.11.15。原有 PyTorch 2.8.0/cu128 和基础锁文件保持不变。
- 训练附加依赖继续由安装脚本安装；固定 PEFT 0.17.1、Faster-Whisper 1.2.1、imageio-ffmpeg 0.6.0、ONNX Runtime 1.23.2 和 protobuf 3.19.6。后两项避免与音频依赖要求的 protobuf 版本冲突；这些附加安装项尚未合并进 uv.lock。
- FFmpeg 从 imageio-ffmpeg 的已安装程序复制到项目本地 `.runtime/bin`，启动时验证并加入 PATH。无需提交二进制程序。
- UTF-8 编码、NLTK 数据与下载缓存使用项目本地设置。
- 默认监听 `127.0.0.1:7860`。可将 `local-settings.example.bat` 复制为 `local-settings.bat`，配置 `INDEXTTS_HOST`、`INDEXTTS_PORT` 和可选 `HTTP_PROXY` / `HTTPS_PROXY`。两个启动脚本都会读取该文件；Windows PowerShell 下载也显式使用配置的代理。
- 未配置代理时沿用调用环境或直接连接。`NO_PROXY` 包含 localhost 与所选监听地址。

新机器先运行 `1- install.bat`，再运行 `2- run.bat`。直接运行 `easy_gui.py` 时，监听地址和端口从同名环境变量读取；批处理本地配置文件由启动脚本负责加载。

`.venv`、`.runtime`、模型、语料、训练项目、输出和本地配置均不提交。已有运行副本无需迁移；本仓库先用于源码开发和追踪，不会自动复制现有环境或模型。

## 简体中文界面

- 默认简体中文；顶部“界面语言 / UI Language”可选择 English，保存后手动重启生效。
- 覆盖声音库、语音合成、对话、数据准备、LoRA 训练、说明和常见状态/错误提示。控制台、原始异常详情和 TensorBoard 保留原文。
- 翻译只改变枚举显示标签，内部值、路径、用户文本与已保存项目字段保持兼容；缺少翻译时回退英文。
- `ui_language` 保存在被忽略的 `.runtime/gui_settings.json` 中，与 CUDA 设置合并写入，保留其他字段。

使用安装后的 Python 运行离线验证：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_easy_gui_i18n.py -v
```

## LoRA 合并导出

`tools/merge_lora_to_pth.py` 使用 PEFT 的原生 CPU 合并，将当前 acoustic-only GPT LoRA 合入训练时的原始 FP32 `gpt.pth`，输出可由 ComfyUI 的 IndexTTS-2.5 GPT Checkpoint Selector 加载的完整 checkpoint。无需 GPU，也不会覆盖已有输出。

```powershell
.\.venv\Scripts\python.exe -B tools/merge_lora_to_pth.py `
  --base models/IndexTTS-2.5/gpt.pth `
  --config models/IndexTTS-2.5/config.yaml `
  --adapter training/outputs/my_voice/final_adapter `
  --output outputs/merged/my_voice.pth
```

可选 `--expected-base <部署用的原始 gpt.pth>` 会核对训练与部署底模的 SHA256。合并要求标准 bias-free LoRA；额外训练模块、DoRA、自定义 rank/alpha pattern 等未验证变体会明确拒绝。输入支持平铺 state_dict 或其外层 `model` 字段。

导出会检查 LoRA 张量配对、所有原始键/形状/精度、非目标权重不变、数值有限、合并前后固定输入的 CPU 前向误差、保存回读和源文件校验值，并在旁边写 `.merge.json` 报告。FP32 合并后再做 BF16 推理，以及随机采样，可能与实时加载 LoRA 有细微差异，音质仍需试听。

输出 `.pth`、LoRA、训练状态和带本机路径的导出报告不纳入 Git。部署时将导出文件放入 ComfyUI 的 `models/IndexTTS-2.5/gpt_checkpoints/`，刷新 checkpoint 选择节点，连接到合成节点的 `gpt_checkpoint` 输入；保留原始底模，继续提供参考音频。

小模型离线验证（无需真实模型、语料、网络或 GPU）：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_merge_lora_to_pth.py -v
```

## 已验证范围

原运行副本使用 Python 3.11.15、PyTorch 2.8.0/cu128，完成中英文界面构建和 HTTP 检查，以及 Rank 4 / Alpha 8 的训练与保存；真实 LoRA 合并通过 456 项权重回读和 ComfyUI 插件严格加载检查。新源码仓库借用既有环境，已通过新增的 8 项单元检查、中英文界面构建、设置/控制回调验证和临时本机 HTTP 200 检查；测试服务检查后关闭。未在新目录重新安装全套环境，也未重新训练。显存需求随片段长度和训练参数变化。
