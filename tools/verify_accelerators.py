import importlib.metadata
import importlib.util
import sys
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__} CUDA: {torch.version.cuda} Available: {torch.cuda.is_available()}")

PACKAGES = (
    ("triton", "triton-windows"),
    ("flash_attn", "flash-attn"),
)

for module_name, distribution_name in PACKAGES:
    try:
        version = importlib.metadata.version(distribution_name)
        found = importlib.util.find_spec(module_name) is not None
        if found:
            print(f"{module_name}: True ({version})")
        else:
            print(f"{module_name}: False ({distribution_name} {version} installed, module not found)")
    except importlib.metadata.PackageNotFoundError:
        print(f"{module_name}: False (not installed)")
    except Exception as exc:
        print(f"{module_name}: False ({type(exc).__name__}: {exc})")
