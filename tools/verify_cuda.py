import sys
import torch

print(f"[Python] {sys.version.split()[0]}")
print(f"[PyTorch] {torch.__version__}")
print(f"[PyTorch CUDA] {torch.version.cuda}")
print(f"[CUDA available] {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

expected_python = (3, 11, 15)
expected_torch = "2.8.0+cu128"
expected_cuda = "12.8"

ok = True
if sys.version_info[:3] != expected_python:
    print(f"[ERROR] Expected Python {'.'.join(map(str, expected_python))}.")
    ok = False
if torch.__version__ != expected_torch:
    print(f"[ERROR] Expected PyTorch {expected_torch}.")
    ok = False
if torch.version.cuda != expected_cuda:
    print(f"[ERROR] Expected PyTorch CUDA runtime {expected_cuda}.")
    ok = False
if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available to PyTorch.")
    ok = False

raise SystemExit(0 if ok else 1)
