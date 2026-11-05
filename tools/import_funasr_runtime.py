"""Import the existing portable CPU SenseVoice runtime, without modifying its source.

Example: python tools/import_funasr_runtime.py --source D:\\Code\\funasr-llamacpp-windows-x64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


MODEL_SHA256 = "4ae45c94422de949b387e2e0fb10d7e14e4c42c69db30c3444ecc7d4b844b7c5"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _copy_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, temp)
        if sha256(source) != sha256(temp):
            raise RuntimeError(f"Copy checksum mismatch: {destination}")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def import_runtime(source, root, replace=False):
    source, root = Path(source).resolve(strict=True), Path(root).resolve()
    executable = source / "llama-funasr-sensevoice.exe"
    model = source / "models/sensevoice-small-q8.gguf"
    if not executable.is_file() or not model.is_file():
        raise ValueError("Source must contain llama-funasr-sensevoice.exe and models/sensevoice-small-q8.gguf")
    if sha256(model) != MODEL_SHA256:
        raise ValueError("SenseVoiceSmall Q8 model SHA-256 mismatch; import aborted")
    with executable.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError("Source executable is not a Windows PE executable")
    runtime_dir = root / ".runtime/funasr"
    pairs = [(executable, runtime_dir / executable.name),
             (model, root / "models/asr" / model.name)]
    # Ship available notices with the private runtime, not its historical logs/work.
    for item in source.iterdir():
        if item.is_file() and (item.name.lower() == "readme.md" or
                               item.name.lower().startswith(("license", "notice", "copying"))):
            pairs.append((item, runtime_dir / item.name))
    hashes = [(src, dst, sha256(src)) for src, dst in pairs]
    for src, dst, digest in hashes:
        if dst.exists() and sha256(dst) != digest and not replace:
            raise ValueError(f"Destination differs: {dst}; use --replace to replace that runtime")
    for src, dst, digest in hashes:
        if not dst.exists() or sha256(dst) != digest:
            _copy_atomic(src, dst)
    receipt = {"source_directory": str(source),
               "upstream": "https://github.com/modelscope/FunASR",
               "backend": "cpu", "model": "SenseVoiceSmall Q8",
               "files": [{"path": dst.relative_to(root).as_posix(), "sha256": digest,
                          "bytes": dst.stat().st_size} for _, dst, digest in hashes]}
    receipt_path = runtime_dir / "import.json"
    temporary = receipt_path.with_suffix(".json." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing portable Windows CPU package directory")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Easy GUI deployment root")
    parser.add_argument("--replace", action="store_true", help="Replace a different previously imported runtime")
    args = parser.parse_args(argv)
    result = import_runtime(args.source, args.root, args.replace)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
