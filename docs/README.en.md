# IndexTTS 2.5 Easy GUI

[简体中文](../README.md) | English

A local Windows GUI for [IndexTTS 2.5](https://github.com/index-tts/index-tts). Clone a voice from reference audio and synthesize speech; you can also prepare your own speech corpus, fine-tune a LoRA, and export the result for ComfyUI.

![English dark interface, using isolated fake-engine demonstration data](<../assets/webui-2-dark.png>)

[See the light/dark theme sweep](<../assets/webui-theme-sweep.png>) (Chinese interface).

## Features

- **Voice Library**: upload or record reference clips and reuse them.
- **Inference**: single-text and multi-speaker dialogue synthesis, automatic long-text chunking, and emotion control from reference audio, an emotion vector, or a text description.
- **Dataset Preparation**: fill in missing transcripts with SenseVoice on the CPU, then proofread each clip by ear.
- **LoRA Training** (experimental): fine-tune the GPT module on your corpus, with checkpoints, resume, early stop, validation loss (including post-hoc evaluation of saved checkpoints), evaluation audio, and TensorBoard.
- **Blind A/B Test**: compare two adapters (or the base model) by blind listening, with preference statistics and a significance test, to tell whether a new LoRA is a real improvement or a regression.
- **Export GPT Checkpoint**: merge a LoRA into a full `.pth` that the ComfyUI IndexTTS-2.5 nodes can load.
- **Updater**: Git-free ZIP updates that keep your data and support rollback.

## Requirements

- Windows 10/11 x64
- An NVIDIA GPU and driver for inference and training; speech recognition and checkpoint merging compute on the CPU
- Internet access for the first install and the first launch

The maintainer used the following hardware. It is a reference point, not a minimum requirement:

| CPU | RAM | GPU |
| --- | --- | --- |
| Intel i7-10700 | 64 GB DDR4 | NVIDIA RTX A3000LP 12 GB (Ampere) |

Memory use depends on clip length, synthesis settings, and training parameters.

## Quick start

1. Download the project and double-click **`1- install.bat`**. The installer puts Python and all dependencies inside the project folder and leaves any system Python alone.
2. Double-click **`2- run.bat`**. The first launch downloads the IndexTTS 2.5 model, then the GUI opens at <http://127.0.0.1:7860>.
3. To use speech recognition under Projects → Corpus, import the ASR runtime and model separately; see [Installation](installation.md#导入-asr-程序和模型).

## Updating

Close the GUI, then double-click **`3- update.bat`**. Models, datasets, training outputs, and the voice library are kept.

```powershell
& '.\3- update.bat'             # update to the latest version
& '.\3- update.bat' --check     # check only
& '.\3- update.bat' --rollback  # restore the previous version
```

If a new version changes dependencies, the updater stops before replacing any files and asks you to rerun the installer.

## Documentation (Chinese)

| Document | Topics |
| --- | --- |
| [安装与配置](installation.md) | Installation, launch, ASR import, listen address and proxy, accelerators, data folders |
| [使用指南](user-guide.md) | Voice library, inference, dataset preparation, LoRA training, blind A/B test, export |
| [更新与常见问题](update-and-troubleshooting.md) | ZIP updates, rollback, troubleshooting |
| [开发与维护](development.md) | Code layout, tests, release manifest, versioning, UI translation |
| [更新日志](CHANGELOG.md) | Changes in each version, including which versions need a reinstall |

## Credits and license

This project is developed and maintained independently, based on [Mixomo/IndexTTS-2.5-Easy-GUI](https://github.com/Mixomo/IndexTTS-2.5-Easy-GUI); the original history, authorship, and license are preserved. It builds on or takes inspiration from:

- [IndexTTS](https://github.com/index-tts/index-tts): the speech model and inference code. See that repository for the model, paper, and benchmarks.
- [FunASR](https://github.com/modelscope/FunASR) / [SenseVoice](https://github.com/FunAudioLLM/SenseVoice): the speech recognition runtime and model.
- [Triton for Windows](https://github.com/triton-lang/triton-windows), [FlashAttention](https://github.com/Dao-AILab/flash-attention), and its [Windows wheels](https://github.com/kingbri1/flash-attention): optional inference accelerators.
- [Voice Clone Studio](https://github.com/FranckyB/Voice-Clone-Studio): design reference for the GUI and workflow.

Use and distribution are subject to the [LICENSE](../LICENSE), the [DISCLAIMER](../DISCLAIMER), and the licenses of each third-party program and model. Changes made in this project are not endorsed or warranted by the original model rights holders.
