# 听音校对与 FunASR 转写

“数据集准备”页内置听音校对工作台。数据集、参考音频库、推理参考音频和训练评估参考音频统一使用 **SenseVoiceSmall Q8 / CPU** 转写。当前集成使用模型的自动语言识别，不提供 Whisper 模型选择、强制语言或批大小选项；训练项目的语言设置仍用于训练清单。

## 日常使用

1. 在 GUI 中创建或选择项目，填写语料目录并保存项目。目录指运行 GUI 的电脑上的路径；通过局域网访问时也填写服务器路径，不是浏览器所在设备的路径。
2. 在校对工作台加载目录。扫描包含子目录，并使用相对路径区分同名音频；支持范围与训练扫描一致。
3. 点击“识别缺失文本”。程序逐条补充缺少或空白的同名 TXT，跳过已有正文。文件搜索和状态筛选不会缩小批量识别范围。
4. 听音并修改正文，点击“保存”或“保存并下一条”。后者同时将条目标记为已校对；校对标记只记录进度，不决定是否进入训练。
5. 点击“更新训练清单”，再开始训练。更新操作读取已保存的 TXT，不再次调用 ASR。语料修改后清单会显示待更新；GUI 管理的项目不能使用过期清单启动训练。

“重新识别当前条目”将结果放入编辑框，**不会自动覆盖 TXT**。检查后再保存。识别失败保留当前编辑内容；批量识别可停止，已完成文本保留，再次启动时会跳过已有正文。批量识别期间不能切换该页面的项目或目录。

快捷键：`Ctrl+S` 保存、`Ctrl+Enter` 保存并下一条、`Alt+Space` 播放或暂停。浏览器占用快捷键时，可使用对应按钮。切换条目或目录前会处理未保存修改；更换 GUI 标签页不会重新创建校对工作台。

## 保存与清单保护

- TXT 保存为 UTF-8，空白正文不能手工保存。音频文件不改写；转码仅用于临时识别和试听。
- 首次修改已有 TXT 前，原文备份到语料目录的 `.proofread/backups/`，保留相对目录结构。后续保存不会覆盖首次备份。
- 校对进度保存在 `.proofread/state.json`，兼容独立工具已有的第一层 WAV 进度。文本变化后，旧校对标记失效。
- 如果其他页面或程序修改了 TXT，旧编辑页面不会直接覆盖新内容。发生冲突时先复制需要保留的编辑内容，再重新加载磁盘文本。
- ASR 落盘前会再次检查 TXT；识别期间被人工填写的条目会跳过。
- 正在运行的训练继续使用其原清单。此时可识别和校对，但更新清单被禁止；等待训练结束后再更新。清单更新失败会保留旧清单。
- 手动指定的外部训练清单继续沿用原有行为，不自动绑定本工作台的语料同步检查。

## 安装或恢复 ASR 运行时

当前 Windows 部署已导入独立运行时。源码仓库不包含 EXE 或模型，其他机器需要准备 Windows CPU 便携包及 SenseVoiceSmall Q8 模型：

- [FunASR 官方运行时发布页](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6)
- [SenseVoiceSmall GGUF 模型](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)

待导入目录需包含 `llama-funasr-sensevoice.exe` 和 `models/sensevoice-small-q8.gguf`。在 Easy GUI 根目录运行：

```powershell
.\.venv\Scripts\python.exe tools\import_funasr_runtime.py --source "D:\Code\funasr-llamacpp-windows-x64"
```

也可用 `--root "D:\IndexTTS-2.5-Easy-GUI-main"` 指定部署根目录。导入器验证模型 SHA-256、检查复制结果，并记录实际文件哈希；若目标已有不同文件，默认停止，只有明确使用 `--replace` 才替换。

导入后的目录：

| 位置 | 用途 |
| --- | --- |
| `.runtime/funasr/` | 识别程序、可用的上游说明及 `import.json` 导入记录 |
| `models/asr/sensevoice-small-q8.gguf` | 本地 Q8 模型 |
| `.runtime/asr/work/` | 每次识别独立的临时音频，任务结束后清理 |
| `.runtime/asr/logs/` | 识别输出、错误及诊断信息 |

默认使用 CPU，单条任务超时为 180 秒。项目已有 FFmpeg 将输入转为 16 kHz 单声道 PCM16；传入识别程序的是英文相对路径，因此语料可以使用中文目录和文件名。所有 GUI ASR 请求串行执行；识别失败不会自动改用 Whisper。底层 tokenizer 等代码仍依赖的 Whisper 包予以保留。

原独立校对工具及原 ASR 工具目录不被删除或改写。模型目录通过 `models/asr/README.md` 保留在版本管理中，GGUF 模型、运行时、日志、语料和 `.proofread` 记录均排除在版本管理之外。

## 验证命令

从项目根目录运行以下检查。Python 测试使用临时样本或模拟进程，不批量修改真实语料；前端状态测试需要 Node.js。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_funasr_asr.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_proofread.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_dataset_proofread.py
node tests/test_proofread_frontend.cjs
```

浏览器验收应检查：局域网访问、播放与拖动、未保存切换提示、批量补缺后刷新、单条重识别不自动保存，以及保存 TXT 后必须更新训练清单。
