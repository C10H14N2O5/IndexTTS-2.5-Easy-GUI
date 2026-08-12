# Copyright (c) 2024 NVIDIA CORPORATION.
# Licensed under the MIT license.

import os
import pathlib

import torch
from torch.utils import cpp_extension


def load():
    """Build BigVGAN's optional fused activation only for the active GPU.

    The upstream loader hard-coded sm_70 and sm_80. CUDA Toolkit 13 removes
    support for compiling compute_70, and compiling architectures unrelated to
    the current card is unnecessary for this single-GPU Easy GUI. The GUI also
    prevents reaching this loader when the external CUDA Toolkit does not match
    the CUDA version used to build PyTorch.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("BigVGAN CUDA kernel requires an NVIDIA CUDA device")
    if not cpp_extension.CUDA_HOME:
        raise RuntimeError("CUDA Toolkit / nvcc was not found")

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    arch = f"{major}{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    buildpath.mkdir(parents=True, exist_ok=True)

    extra_cuda_flags = [
        "-O3",
        "-gencode",
        f"arch=compute_{arch},code=sm_{arch}",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    sources = [
        srcpath / "anti_alias_activation.cpp",
        srcpath / "anti_alias_activation_cuda.cu",
    ]
    return cpp_extension.load(
        name="anti_alias_activation_cuda",
        sources=sources,
        build_directory=buildpath,
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_flags,
        verbose=True,
    )
