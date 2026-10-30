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
