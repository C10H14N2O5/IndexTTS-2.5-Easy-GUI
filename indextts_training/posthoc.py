"""Validation loss for saved LoRA checkpoints after training has finished.

The runtime is loaded exactly like :func:`indextts_training.trainer.train_lora`
loads it (BF16, no accelerated inference paths) and every checkpoint is scored
with the trainer's own ``_validation_loss``. On the same validation manifest
the values therefore match the losses logged during training, up to BF16
numerical noise.
"""
from __future__ import annotations

import gc
from pathlib import Path


class _CancelFlag:
    """Adapter that lets a plain callable stand in for a ``threading.Event``."""

    def __init__(self, cancelled):
        self._cancelled = cancelled

    def is_set(self):
        return bool(self._cancelled())


def evaluate_checkpoints(model_dir, targets, rows, *, on_progress=None, on_result=None,
                         cancelled=lambda: False, seed=1234):
    """Score ``targets`` in order on ``rows``.

    ``targets`` is a list of ``{"name": str, "path": str | None, "step": int | None}``;
    ``path=None`` is the base model without an adapter. ``on_result(target, loss)``
    is called as soon as one target finishes, so a cancelled run keeps earlier
    results. ``on_progress(done, total, target, sample, samples)`` reports
    per-sample progress. Returns ``{"completed": [names], "stopped": bool}``.
    """
    import torch
    from peft import PeftModel
    from indextts.infer_v2_5 import IndexTTS2
    from indextts_training import trainer

    rows = tuple(dict(row) for row in rows)
    if not rows:
        raise RuntimeError('Validation manifest is empty.')
    if not torch.cuda.is_available():
        raise RuntimeError('IndexTTS 2.5 LoRA training requires an NVIDIA CUDA device.')
    model_dir = Path(model_dir)
    if not (model_dir / 'config.yaml').is_file():
        raise RuntimeError(f'IndexTTS 2.5 model config not found: {model_dir / "config.yaml"}')
    stop = _CancelFlag(cancelled)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    device = 'cuda:0'
    units = max(1, len(targets) * len(rows))
    completed = []
    tts = IndexTTS2(cfg_path=str(model_dir / 'config.yaml'), model_dir=str(model_dir), device=device,
                    use_bf16=True, use_cuda_kernel=False, use_gpt_latent=False, use_torch_compile=False,
                    use_deepspeed=False, use_accel=False, use_qwen_emo=False)
    try:
        trainer._freeze_index_tts_modules(tts)
        base = tts.gpt.gpt
        wrapped = None
        active = None
        for index, target in enumerate(targets):
            if stop.is_set():
                break

            def progress(sample, count, index=index, target=target):
                if on_progress is not None:
                    on_progress(index * len(rows) + sample, units, target, sample, count)

            path = target.get('path')
            step = int(target.get('step') or 0)
            if path is None:
                if wrapped is None:
                    tts.gpt.gpt = base
                    loss = trainer._validation_loss(tts, rows, device, step, stop, on_sample=progress)
                else:
                    with wrapped.disable_adapter():
                        loss = trainer._validation_loss(tts, rows, device, step, stop, on_sample=progress)
            else:
                name = f'posthoc_{index}'
                if wrapped is None:
                    wrapped = PeftModel.from_pretrained(base, str(path), adapter_name=name, is_trainable=False)
                    tts.gpt.gpt = wrapped
                else:
                    wrapped.load_adapter(str(path), adapter_name=name, is_trainable=False)
                    wrapped.set_adapter(name)
                    if active is not None:
                        wrapped.delete_adapter(active)
                active = name
                loss = trainer._validation_loss(tts, rows, device, step, stop, on_sample=progress)
            if loss is None:
                break
            completed.append(target['name'])
            if on_result is not None:
                on_result(target, float(loss))
            if on_progress is not None:
                on_progress((index + 1) * len(rows), units, target, len(rows), len(rows))
    finally:
        del tts
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return {'completed': completed, 'stopped': stop.is_set()}
