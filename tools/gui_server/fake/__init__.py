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


def _write_adapter(directory: Path, config: dict, step: int = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    adapter_config = {"peft_type": "LORA", "bias": "none", "r": int(config.get("rank", 16)),
                      "lora_alpha": int(config.get("alpha", 32)), "lora_dropout": float(config.get("dropout", 0.05)),
                      "target_modules": ["c_attn", "c_proj", "c_fc"], "task_type": None}
    (directory / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")
    # Distinct weights per step, so the final adapter matches the last checkpoint exactly.
    (directory / "adapter_model.safetensors").write_bytes(f"FAKE-SAFETENSORS-{int(step)}".encode("ascii"))


FAKE_AUTO_STOP_FILE = "fake_auto_stop.json"  # stands in for trainer_state.pt["auto_stop"]


def fake_train_lora(config, callback, stop_event):
    """Stand-in for ``indextts_training.trainer.train_lora`` with real TensorBoard events."""
    from torch.utils.tensorboard import SummaryWriter
    from indextts_training.auto_stop import monitor_from_config
    from indextts_training.schedule import (checkpoint_due as due_checkpoint, read_adapter_state, validation_due,
                                            write_adapter_state)

    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = max(1, _manifest_rows(config.get("manifest")))
    per_epoch = max(1, math.ceil(rows / (int(config.get("batch", 1)) * int(config.get("grad_accum", 8)))))
    total = int(config.get("steps", 100)) if config.get("mode") == "Steps" else int(config.get("epochs", 2)) * per_epoch
    end_step = int(config.get("end_step") or total)
    if end_step < 1 or end_step > total:
        raise RuntimeError(f"Early stop step {end_step} must be between 1 and the planned {total} steps.")
    save_every = max(1, int(config.get("save_every", 50)))
    resume = str(config.get("resume") or "")
    sidecar = read_adapter_state(resume) if resume else None
    start = sidecar["step"] if sidecar else (
        int(resume.rsplit("-", 1)[-1]) if resume and resume.rsplit("-", 1)[-1].isdigit() else 0)
    if start >= end_step:
        raise RuntimeError(f"Resume checkpoint is already at step {start}, at or beyond the end step {end_step}.")
    validation = config.get("validation") or None
    interval = int((validation or {}).get("interval_steps") or 0)
    dense = (validation or {}).get("dense")
    monitor = monitor_from_config(config, (validation or {}).get("fingerprint"))
    if monitor is not None and not validation:
        raise RuntimeError("Auto stop needs validation loss. Enable validation for this project first.")
    if monitor is not None and resume:
        try:
            saved = json.loads((Path(resume) / FAKE_AUTO_STOP_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = None
        if saved and monitor.restore(saved):
            callback(message=f"Auto stop state restored from step {start}.")
    overfit = "overfit" in out.name.lower()
    auto_stopped = False

    def save(directory: Path, saved_step: int, extra: dict | None = None) -> None:
        _write_adapter(directory, config, saved_step)
        write_adapter_state(directory, saved_step, total, end_step, stop_event.is_set(), extra)
        if monitor is not None:
            (directory / FAKE_AUTO_STOP_FILE).write_text(json.dumps(monitor.to_dict()), encoding="utf-8")
    delay = max(0.02, min(0.25, 40.0 / max(1, end_step - start)))
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
        for step in range(start + 1, end_step + 1):
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
            eta = elapsed / max(1, step - start) * (end_step - step)
            plan = f" (LR schedule {total})" if end_step != total else ""
            callback(step=step, total=end_step, schedule_total=total, epoch=epoch, loss=ema, elapsed=elapsed, eta=eta,
                     message=f"Training step {step}/{end_step}{plan} | epoch {epoch} | lr {lr:.2e}")
            checkpoint_due = due_checkpoint(step, save_every, end_step)
            if validation and (validation_due(step, end_step=end_step, save_every=save_every, interval=interval,
                                              dense=dense) or (monitor is not None and monitor.extra_validation_due(step))):
                callback(message=f"Running validation at step {step}…")
                if overfit:
                    # A U-shaped curve: improves until 40 % of the plan, then overfits.
                    value = 0.9 + 2.5 * (progress - 0.4) ** 2 + 0.002 * math.sin(step * 0.9)
                else:
                    value = ema * 1.12 + 0.05
                writer.add_scalar("validation/loss", value, step)
                callback(message=f"Validation loss at step {step}: {value:.6f}")
                if monitor is not None:
                    event = monitor.observe(step, value)
                    snapshot = monitor.snapshot()
                    writer.add_scalar("auto_stop/best_loss", snapshot["best_loss"], step)
                    writer.add_scalar("auto_stop/valley_loss", snapshot["valley_loss"], step)
                    writer.add_scalar("auto_stop/watching", 0.0 if snapshot["phase"] == "normal" else 1.0, step)
                    if event["new_best"]:
                        save(out / "best_adapter", step, {"kind": "best", "validation_loss": float(value)})
                        callback(message=f"Saved best adapter at step {step} (validation loss {value:.6f}).")
                    if event["entered_watch"]:
                        callback(message=f"Auto stop: no significant improvement at step {step}; validating every "
                                         f"{monitor.watch_interval_steps} steps until step {snapshot['deadline_step']}.")
                    if event["recovered"]:
                        callback(message=f"Auto stop: validation loss improved to {value:.6f} at step {step}; "
                                         f"resuming normal validation.")
                    if event["stop"]:
                        auto_stopped = True
                        checkpoint_due = True
                    callback(auto_stop=snapshot)
            if checkpoint_due:
                checkpoint = out / f"checkpoint-step-{step:06d}"
                save(checkpoint, step, {"auto_stopped": True} if auto_stopped else None)
                eval_name = ""
                if config.get("eval_enabled") and config.get("eval_text") and config.get("eval_reference"):
                    eval_wav = write_tone(out / "evaluation" / f"eval-step-{step:06d}.wav", 2.0, 200 + step % 200)
                    eval_name = eval_wav.name
                callback(message=f"Saved checkpoint: {checkpoint.name}" + (f" | Eval: {eval_name}" if eval_name else ""))
            writer.flush()
            if auto_stopped:
                break
    finally:
        writer.close()
    final = out / "final_adapter"
    stopped = stop_event.is_set()
    save(final, step, {"auto_stopped": True} if auto_stopped else None)
    result = {"step": step, "total": total, "schedule_total": total, "end_step": end_step, "output": str(final),
              "stopped": stopped, "early_stopped": not stopped and not auto_stopped and end_step < total and step == end_step,
              "auto_stopped": auto_stopped and not stopped}
    if monitor is not None:
        snapshot = monitor.snapshot()
        result.update(best_step=snapshot["best_step"], best_loss=snapshot["best_loss"], valley_step=snapshot["valley_step"],
                      watch_start_step=snapshot["watch_start_step"], min_delta=monitor.min_delta,
                      patience_steps=monitor.patience_steps, auto_stop=snapshot)
    return result


def fake_evaluate_checkpoints(model_dir, targets, rows, *, on_progress=None, on_result=None,
                              cancelled=lambda: False, seed=1234):
    """Stand-in for ``indextts_training.posthoc.evaluate_checkpoints``.

    Loss decreases with the checkpoint step; the base model scores highest. A
    checkpoint name containing ``fail`` raises once it is reached.
    """
    rows = list(rows)
    if not rows:
        raise RuntimeError("Validation manifest is empty.")
    units = max(1, len(targets) * len(rows))
    completed = []
    time.sleep(0.3)  # model load
    for index, target in enumerate(targets):
        for sample in range(len(rows)):
            if cancelled():
                return {"completed": completed, "stopped": True}
            if on_progress is not None:
                on_progress(index * len(rows) + sample, units, target, sample, len(rows))
            time.sleep(0.03)
        if "fail" in str(target["name"]).lower():
            raise RuntimeError(f"Fake validation failure requested by {target['name']}.")
        step = target.get("step")
        loss = 2.9 if target.get("path") is None else 0.9 + 1.6 * math.exp(-float(step or 0) / 300.0)
        completed.append(target["name"])
        if on_result is not None:
            on_result(target, loss)
    return {"completed": completed, "stopped": bool(cancelled())}


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
    state.extras["posthoc_fn"] = fake_evaluate_checkpoints
    state.extras["model_dir"] = model_dir()
    print("[fake-engine] INDEXTTS_FAKE_ENGINE=1: synthesis, training, ASR and export are simulated.", flush=True)
