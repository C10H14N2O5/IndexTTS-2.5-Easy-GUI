"""Merge an acoustic-only Easy GUI GPT LoRA into a canonical FP32 checkpoint.

Uses PEFT's native merge on CPU, keeps every original checkpoint key, and
never overwrites the base model, adapter or an existing export.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2Model


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def zero_positions(positions, dim):
    return torch.zeros((*positions.shape, dim), device=positions.device)


def make_gpt(state, config):
    """Construct just the canonical transformer, without inference wrappers."""
    model_dim = state["gpt.ln_f.weight"].numel()
    layers = len({key.split(".")[2] for key in state if key.startswith("gpt.h.")})
    positions = state["mel_pos_embedding.emb.weight"].shape[0] + state["text_pos_embedding.emb.weight"].shape[0]
    hf_config = GPT2Config(vocab_size=256, n_positions=positions, n_ctx=positions,
                          n_embd=model_dim, n_layer=layers, n_head=int(config["heads"]),
                          use_cache=False, attn_implementation="eager")
    with torch.device("meta"):
        model = GPT2Model(hf_config)
    del model.wte
    del model.wpe
    model.wpe = functools.partial(zero_positions, dim=model_dim)
    trunk = {key.removeprefix("gpt."): value for key, value in state.items() if key.startswith("gpt.")}
    model.load_state_dict(trunk, strict=True, assign=True)
    # Nonpersistent causal-mask buffers are not present in a state_dict.
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type != "meta":
            continue
        if name.endswith(".attn.bias"):
            replacement = torch.tril(torch.ones(buffer.shape, dtype=buffer.dtype))
        elif name.endswith(".attn.masked_bias"):
            replacement = torch.tensor(-1e4, dtype=buffer.dtype)
        else:
            raise RuntimeError(f"Unknown nonpersistent buffer: {name}")
        parent, _, leaf = name.rpartition(".")
        model.get_submodule(parent)._buffers[leaf] = replacement
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-base", type=Path, help="Verify the deployment base is identical")
    args = parser.parse_args()
    base_path, adapter, output = args.base.resolve(), args.adapter.resolve(), args.output.resolve()
    report_path = output.with_suffix(".merge.json")
    pending = output.with_suffix(output.suffix + ".partial")
    if output.exists() or report_path.exists() or pending.exists():
        raise FileExistsError(f"Export already exists: {output}")
    if output == base_path or output.is_relative_to(adapter):
        raise ValueError("Output must be separate from the base and adapter")
    adapter_weights = adapter / "adapter_model.safetensors"
    adapter_config_path = adapter / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("bias") != "none":
        raise ValueError("Expected a standard, bias-free acoustic GPT LoRA")
    if any(adapter_config.get(key) for key in ("modules_to_save", "use_dora", "use_rslora", "rank_pattern", "alpha_pattern", "lora_bias")):
        raise ValueError("This exporter accepts the tested standard acoustic-only LoRA profile")
    started = time.time()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    print("Hashing and checking base/adapter files...", flush=True)
    base_hash = digest(base_path)
    if args.expected_base and digest(args.expected_base.resolve()) != base_hash:
        raise ValueError("Training base and deployment base differ")
    adapter_hash = digest(adapter_weights)
    adapter_config_hash = digest(adapter_config_path)
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("model", checkpoint)
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected a tensor state dictionary")
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("This export expects the original FP32 base checkpoint")
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["gpt"]
    keys = set(state)
    lora_state = load_file(str(adapter_weights), device="cpu")
    targets = set()
    for key in lora_state:
        if not key.startswith("base_model.model.") or not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"Unexpected adapter tensor (would not be exported): {key}")
        prefix = key.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]
        a, b = f"base_model.model.{prefix}.lora_A.weight", f"base_model.model.{prefix}.lora_B.weight"
        if a not in lora_state or b not in lora_state or f"gpt.{prefix}.weight" not in state:
            raise ValueError(f"Incomplete or unmatched LoRA pair: {prefix}")
        targets.add(f"gpt.{prefix}.weight")
    if len(lora_state) != 2 * len(targets):
        raise ValueError("Adapter tensor coverage mismatch")
    print(f"Building CPU transformer: {len(keys)} checkpoint tensors, {len(targets)} LoRA targets...", flush=True)
    model = make_gpt(state, config)
    wrapped = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True).eval()
    generator = torch.Generator().manual_seed(20261031)
    probe = torch.randn(1, 24, model.config.n_embd, generator=generator)
    with torch.inference_mode():
        unmerged_output = wrapped(inputs_embeds=probe, use_cache=False, return_dict=True).last_hidden_state.clone()
        merged = wrapped.merge_and_unload(safe_merge=True).eval()
        merged_output = merged(inputs_embeds=probe, use_cache=False, return_dict=True).last_hidden_state
    torch.testing.assert_close(merged_output, unmerged_output, rtol=2e-4, atol=2e-4)
    max_difference = float((merged_output - unmerged_output).abs().max())
    trunk = {f"gpt.{key}": value.detach() for key, value in merged.state_dict().items()}
    if set(trunk) != {key for key in keys if key.startswith("gpt.")}:
        raise ValueError("Merged transformer keys do not match the canonical base")
    result = dict(state)
    result.update(trunk)
    changed = []
    print("Checking all tensor names, shapes, dtypes and merge results...", flush=True)
    for key in sorted(keys):
        tensor = result[key]
        original = state[key]
        if tensor.shape != original.shape or tensor.dtype != original.dtype:
            raise ValueError(f"Shape or dtype changed: {key}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite export tensor: {key}")
        if not torch.equal(tensor, original):
            changed.append(key)
            if key not in targets:
                raise ValueError(f"Non-LoRA tensor changed: {key}")
    if not changed:
        raise ValueError("No base tensors changed during merge")
    if any("lora_" in key or "base_model." in key or "inference_model." in key for key in result):
        raise ValueError("PEFT/inference wrapper keys leaked into the output")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {**checkpoint, "model": result} if "model" in checkpoint else result
    print(f"Saving independent FP32 checkpoint: {output}", flush=True)
    try:
        torch.save(payload, pending)
        reloaded = torch.load(pending, map_location="cpu", weights_only=True, mmap=True)
        reloaded_state = reloaded.get("model", reloaded)
        if set(reloaded_state) != keys:
            raise ValueError("Reloaded export has incorrect keys")
        for key in keys:
            if not torch.equal(reloaded_state[key], result[key]):
                raise ValueError(f"Saved tensor did not round-trip: {key}")
        del reloaded, reloaded_state
        if digest(base_path) != base_hash or digest(adapter_weights) != adapter_hash or digest(adapter_config_path) != adapter_config_hash:
            raise RuntimeError("A source file changed while exporting")
        pending.rename(output)
    except Exception:
        if pending.exists():
            pending.unlink()
        raise
    report = {
        "base": str(base_path), "base_sha256": base_hash,
        "adapter": str(adapter), "adapter_sha256": adapter_hash,
        "output": str(output), "output_sha256": digest(output),
        "dtype": "float32", "tensor_count": len(keys),
        "lora_tensor_count": len(lora_state), "lora_target_count": len(targets),
        "changed_tensor_count": len(changed), "changed_tensor_names": changed,
        "rank": adapter_config["r"], "alpha": adapter_config["lora_alpha"],
        "cpu_forward_max_abs_difference": max_difference,
        "source_files_unchanged": True, "round_trip_tensors_identical": True,
        "elapsed_seconds": round(time.time() - started, 2),
        "note": "Standard PEFT FP32 merge; BF16 inference and stochastic decoding can produce small differences from a live adapter.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "changed_tensor_names"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
