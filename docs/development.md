# 开发与维护

本文写给修改代码或发布新版本的维护者。普通使用请看 [使用指南](user-guide.md)。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `easy_gui.py` | Gradio 界面和各页面的回调 |
| `indextts_training/` | 数据集与训练清单、听音校对后端、ASR 调用、LoRA 训练、检查点导出、盲测对比（`abx.py`，仅用标准库）、任务互斥 |
| `tools/proofread/` | 内嵌听音校对工作台的前端页面 |
| `tools/` | 安装、启动、更新脚本，环境检查（`verify_*.py`），ASR 导入，LoRA 合并，界面翻译 |
| `indextts/` | 上游 IndexTTS 推理代码（含本项目的少量兼容修改） |
| `backends/trt/` | 上游 TensorRT 后端，Easy GUI 不使用，说明见其自带的 README |
| `1- install.bat`、`2- run.bat`、`3- update.bat` | 用户入口，实际逻辑在 `tools/` 中 |

仓库中还保留着上游的 `indextts` / `indextts2` 命令行（`pyproject.toml` 中注册，测试在 `cli_tests/`）。`indextts2` 只支持 IndexTTS-2 模型，不属于 Easy GUI 支持的用法，也没有对应文档。

## 运行测试

在已安装环境的项目根目录运行。以下测试使用临时文件或模拟进程，不需要真实模型、GPU 或网络，也不会改动真实语料：

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -B -m unittest discover -s tests -p test_easy_gui_i18n.py       # 界面翻译
& $py -B -m unittest discover -s tests -p "test_updat*.py"            # ZIP 更新器
& $py -B -m unittest discover -s tests -p test_runtime_lock.py        # 运行互斥
& $py -B -m unittest discover -s tests -p test_funasr_asr.py          # ASR 调用
& $py -B -m unittest discover -s tests -p "test_*proofread.py"        # 校对与训练清单
& $py -B -m unittest discover -s tests -p "test_*export*.py"          # 检查点导出
& $py -B -m unittest discover -s tests -p test_merge_lora_to_pth.py   # LoRA 合并
& $py -B -m unittest discover -s tests -p test_text_chunking.py       # 长文本分段
& $py -B -m unittest discover -s tests -p test_abx.py                 # 盲测对比
node tests/test_proofread_frontend.cjs                                # 校对前端状态，需要 Node.js
```

`tests/test_v1.py`、`test_v2.py`、`regression_test.py`、`padding_test.py` 来自上游，需要下载模型并使用 GPU。

改动校对工作台后，建议在浏览器中再手动检查：局域网访问、播放与拖动进度、切换时的未保存提示、批量识别后的刷新、单条重识别不自动保存，以及保存 TXT 后训练清单显示“待更新”。

## 发布清单

ZIP 更新器只认 `tools/update-manifest.json` 中列出的文件。它从 GitHub 上 `C10H14N2O5/IndexTTS-2.5-Easy-GUI` 的 `main` 分支下载新版本，所以**推送到 GitHub 的 main 就等于发布**。

**增删程序文件后，要在同一个提交里重新生成清单：**

```powershell
git add -A                 # 让新增和删除的文件反映到 git ls-files 中
git ls-files > "$env:TEMP\files.txt"
python tools\build_update_manifest.py --files-from "$env:TEMP\files.txt"
```

注意事项：

- `files` 中不会包含模型权重、`.runtime/`、`.venv/` 等受保护的路径；放在根目录的文件只有位于 `tools/update_core.py` 的 `ALLOWED_ROOT` 白名单中才会被收录，其余会被**直接跳过**。新的文档请放在 `docs/` 下。
- `environment` 列出的是环境相关文件：`pyproject.toml`、`uv.lock`、`.python-version`、`1- install.bat`、`tools/install_runtime.bat`、`tools/install.ps1`、`tools/verify_*.py`、`tools/nemo_tn*.py`。**这些文件只要有任何改动，所有用户的 ZIP 更新都会被拦住**，用户需要重新下载并运行安装程序。没有必要时不要碰它们，即使只是改一行提示文字（例如 `tools/install_runtime.bat` 里过时的 `run.bat` 字样，也因此暂时保留）。
- 修改 `ALLOWED_ROOT` 等校验规则时，要考虑到用户机器上运行的是**旧版**更新器。

## 界面翻译

界面文字在代码中以英文书写，通过 `t('...')` 翻译。中文译文在 `tools/i18n/locale/easy_gui_zh_CN.json` 中，**键就是英文原文**：

- 修改 `t()` 中的英文原文时，要同步修改 JSON 中对应的键，否则会退回显示英文；
- `{0}` 等占位符必须与原文一致，`test_easy_gui_i18n.py` 会检查；
- 后端日志的翻译模板在 `easy_gui_backend_templates.json` 中；控制台、原始异常详情和 TensorBoard 不翻译；
- 校对工作台的文字在 `tools/proofread/index.html` 中单独维护。

## 环境说明

- 安装器使用 uv 0.13.6 安装锁定的环境。训练附加依赖（PEFT 0.17.1、Faster-Whisper 1.2.1、imageio-ffmpeg 0.6.0、ONNX Runtime 1.23.2、protobuf 3.19.6）由 `tools/install_runtime.bat` 单独安装，尚未并入 `uv.lock`。
- FFmpeg 从 imageio-ffmpeg 复制到 `.runtime/bin`，启动时加入 PATH，不提交二进制文件。
- Windows 下的 NeMo 文本规范化运行在独立的 Conda 环境（`.runtime/nemo-tn`）中，失败时回退到 num2words。
- 直接运行 `easy_gui.py` 时，监听地址和端口从 `INDEXTTS_HOST` / `INDEXTTS_PORT` 环境变量读取，`local-settings.bat` 只由批处理脚本加载。

## 已验证范围

在 Python 3.11.15、PyTorch 2.8.0+cu128 环境下验证过：中英文界面的构建与 HTTP 访问；Rank 4 / Alpha 8 的训练与保存；对真实 LoRA 做合并，回读 456 项权重，并在 ComfyUI 插件中严格加载。显存需求随片段长度和训练参数变化，没有做系统性的测试。
