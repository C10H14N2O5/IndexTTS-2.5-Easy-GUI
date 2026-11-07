# 本地 ASR 模型 / Local ASR models

此目录用于存放 `sensevoice-small-q8.gguf`。目录说明纳入版本管理，模型文件由 `.gitignore` 排除，不应提交到 Git。

This folder is retained in Git through this README. The GGUF model is installed locally and is **not version-controlled**.

- 模型 / Model: SenseVoiceSmall Q8
- 来源 / Source: [FunAudioLLM/SenseVoiceSmall-GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)
- 文件 / Filename: `sensevoice-small-q8.gguf`
- SHA-256: `4ae45c94422de949b387e2e0fb10d7e14e4c42c69db30c3444ecc7d4b844b7c5`
- 运行方式 / Backend: CPU, automatic language detection

识别程序位于项目的 `.runtime/funasr/`，不放在此模型目录内。导入方法见 [导入 ASR 程序和模型](../../docs/installation.md#导入-asr-程序和模型)，使用说明见 [数据准备](../../docs/user-guide.md#数据准备)。
