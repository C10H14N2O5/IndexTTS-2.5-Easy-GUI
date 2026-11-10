"""Development-only fake engine (``INDEXTTS_FAKE_ENGINE=1``).

Replaces IndexTTS2, the LoRA trainer, SenseVoice ASR and the checkpoint merge
script with fast stand-ins so the whole web UI can be developed and tested on a
machine without an NVIDIA GPU or downloaded models. Real data directories are
still used. Failure injection:

* synthesis text containing ``[fail]`` raises inside the fake model;
* an adapter name containing ``fail`` makes fake training fail at step 5;
* an export filename containing ``fail`` makes the fake merge exit non-zero.
"""
from __future__ import annotations

import json
import math
import struct
import threading
import time
import wave
from pathlib import Path

from .. import paths

SAMPLE_RATE = 22050


def model_dir() -> Path:
    """Resolve per application/test root; never retain a real-data path at import."""
    return paths.RUNTIME_DIR / "fake-engine" / "model"


def write_tone(path: Path | str, seconds: float, frequency: float = 220.0, amplitude: float = 0.25) -> Path:
    """Write a 16-bit mono WAV with a soft sine tone and short fades."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(SAMPLE_RATE * max(0.05, seconds)))
    fade = int(SAMPLE_RATE * 0.02)
    data = bytearray()
    for i in range(frames):
        envelope = min(1.0, i / fade if fade else 1.0, (frames - i) / fade if fade else 1.0)
        vibrato = 1.0 + 0.01 * math.sin(2 * math.pi * 5 * i / SAMPLE_RATE)
        value = amplitude * envelope * math.sin(2 * math.pi * frequency * vibrato * i / SAMPLE_RATE)
        data += struct.pack("<h", int(value * 32767))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(bytes(data))
    return path


class FakeTTS:
    """Same ``infer`` signature as IndexTTS2; writes a tone whose length follows the text."""

    def __init__(self, options: tuple):
        self.options = options
        time.sleep(0.6)  # model load
        print(f"[fake-engine] Loaded fake IndexTTS 2.5 with options {options}", flush=True)

    def infer(self, spk_audio_prompt, text, output_path, lang, emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None, use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0,
              duration_factor=1.0, text_normalization=True, reference_text=None, **generation_kwargs):
        text = str(text or "")
        if "[fail]" in text:
            raise RuntimeError("Fake engine failure requested by [fail] in the text.")
        if not spk_audio_prompt or not Path(spk_audio_prompt).is_file():
            raise FileNotFoundError(f"Reference audio not found: {spk_audio_prompt}")
        seconds = min(20.0, max(0.6, 0.07 * len(text) * float(duration_factor or 1.0)))
        time.sleep(min(3.0, 0.3 + 0.015 * len(text)))
        pitch = 180.0 + (sum(map(ord, Path(spk_audio_prompt).stem)) % 120)
        if emo_vector:
            pitch += 60.0 * float(emo_vector[0] or 0.0)
        write_tone(output_path, seconds, pitch)
        print(f"[fake-engine] {lang} | {len(text)} chars -> {seconds:.2f}s | gpt_gen_time: 0.10 | rtf: 0.05", flush=True)
        return output_path


class FakeAsrService:
    """Mimics ``indextts_training.asr.AsrService`` (status/transcribe)."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.model = self.root / "models/asr/sensevoice-small-q8.gguf"
        self.executable = self.root / ".runtime/funasr/llama-funasr-sensevoice.exe"
        self._lock = threading.Lock()

    def status(self):
        return {"available": True, "message": "SenseVoiceSmall Q8 · CPU · fake engine", "backend": "cpu",
                "model": str(self.model), "executable": str(self.executable)}

    def transcribe(self, audio_path, cancel_event=None):
        from indextts_training.asr import AsrCancelled, AsrError
        with self._lock:
            for _ in range(8):
                if cancel_event is not None and cancel_event.is_set():
                    raise AsrCancelled("识别已取消。")
                time.sleep(0.1)
        name = Path(audio_path).stem
        if "fail" in name.lower():
            raise AsrError("模拟识别失败（文件名包含 fail）。")
        return f"这是对 {name} 的模拟识别结果。"


def _manifest_rows(path: str | None) -> int:
    try:
        return sum(1 for line in Path(str(path)).read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _write_adapter(directory: Path, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    adapter_config = {"peft_type": "LORA", "bias": "none", "r": int(config.get("rank", 16)),
                      "lora_alpha": int(config.get("alpha", 32)), "lora_dropout": float(config.get("dropout", 0.05)),
                      "target_modules": ["c_attn", "c_proj", "c_fc"], "task_type": None}
    (directory / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"FAKE-SAFETENSORS")


def fake_train_lora(config, callback, stop_event):
    """Stand-in for ``indextts_training.trainer.train_lora`` with real TensorBoard events."""
    from torch.utils.tensorboard import SummaryWriter

    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = max(1, _manifest_rows(config.get("manifest")))
    per_epoch = max(1, math.ceil(rows / (int(config.get("batch", 1)) * int(config.get("grad_accum", 8)))))
    total = int(config.get("steps", 100)) if config.get("mode") == "Steps" else int(config.get("epochs", 2)) * per_epoch
    save_every = max(1, int(config.get("save_every", 50)))
    resume = str(config.get("resume") or "")
    start = int(resume.rsplit("-", 1)[-1]) if resume and resume.rsplit("-", 1)[-1].isdigit() else 0
    validation = config.get("validation") or None
    interval = int((validation or {}).get("interval_steps") or 0)
    delay = max(0.02, min(0.25, 40.0 / max(1, total)))
    fail = "fail" in out.name.lower()
    writer = SummaryWriter(str(out / "tensorboard"))
    callback(message="Loading IndexTTS 2.5 training runtime…")
    time.sleep(0.8)
    callback(message=f"LoRA trainable parameters: {1234567:,}")
    began = time.time()
    lr_max = float(config.get("lr", 4e-5))
    ema = None
    step = start
    try:
        for step in range(start + 1, total + 1):
            if stop_event.is_set():
                step -= 1
                break
            time.sleep(delay)
            if fail and step == 5:
                raise RuntimeError("Fake training failure requested by the adapter name.")
            progress = step / total
            raw = 2.2 * math.exp(-3.0 * progress) + 0.35 + 0.08 * math.sin(step * 1.7)
            ema = raw if ema is None else 0.9 * ema + 0.1 * raw
            warmup = max(20, min(100, round(total * 0.05)))
            lr = lr_max * (step / warmup if step < warmup else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
            epoch = (step - 1) // per_epoch + 1
            writer.add_scalar("train/loss_ema", ema, step)
            writer.add_scalar("train_raw/loss", raw, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/mel_loss", raw * 0.8, step)
            writer.add_scalar("train/grad_norm", 0.4 + 0.1 * math.cos(step), step)
            writer.add_scalar("train/epoch", epoch, step)
            elapsed = time.time() - began
            eta = elapsed / max(1, step - start) * (total - step)
            callback(step=step, total=total, epoch=epoch, loss=ema, elapsed=elapsed, eta=eta,
                     message=f"Training step {step}/{total} | epoch {epoch} | lr {lr:.2e}")
            checkpoint_due = step % save_every == 0 or step == total
            if validation and ((interval and step % interval == 0) or (not interval and checkpoint_due)):
                callback(message=f"Running validation at step {step}…")
                value = ema * 1.12 + 0.05
                writer.add_scalar("validation/loss", value, step)
                callback(message=f"Validation loss at step {step}: {value:.6f}")
            if checkpoint_due:
                checkpoint = out / f"checkpoint-step-{step:06d}"
                _write_adapter(checkpoint, config)
                eval_name = ""
                if config.get("eval_enabled") and config.get("eval_text") and config.get("eval_reference"):
                    eval_wav = write_tone(out / "evaluation" / f"eval-step-{step:06d}.wav", 2.0, 200 + step % 200)
                    eval_name = eval_wav.name
                callback(message=f"Saved checkpoint: {checkpoint.name}" + (f" | Eval: {eval_name}" if eval_name else ""))
            writer.flush()
    finally:
        writer.close()
    final = out / "final_adapter"
    _write_adapter(final, config)
    return {"step": step, "total": total, "output": str(final), "stopped": stop_event.is_set()}


def _ensure_fake_model() -> None:
    directory = model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "gpt.pth").is_file():
        (directory / "gpt.pth").write_bytes(b"FAKE-GPT-BASE")
    if not (directory / "config.yaml").is_file():
        (directory / "config.yaml").write_text("fake: true\n", encoding="utf-8")


def install(state) -> None:
    """Swap the production singletons for fakes (called by create_app)."""
    from indextts_training.export import ExportManager

    _ensure_fake_model()
    state.asr = FakeAsrService(paths.ROOT)
    state.export = ExportManager(paths.ROOT, script=Path(__file__).with_name("fake_merge.py"))
    state.training.train_fn = fake_train_lora
    state.extras["model_dir"] = model_dir()
    print("[fake-engine] INDEXTTS_FAKE_ENGINE=1: synthesis, training, ASR and export are simulated.", flush=True)
