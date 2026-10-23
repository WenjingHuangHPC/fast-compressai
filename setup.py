"""Optional CUDA extension build; CPU-only installs remain pure Python."""

from __future__ import annotations

import os
from setuptools import setup

ext_modules = []
cmdclass = {}
if os.environ.get("FAST_COMPRESSAI_BUILD_CUDA") == "1":
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except Exception as exc:
        raise RuntimeError("FAST_COMPRESSAI_BUILD_CUDA=1 requires a CUDA-enabled PyTorch installation") from exc
    if CUDA_HOME is None:
        raise RuntimeError("FAST_COMPRESSAI_BUILD_CUDA=1 requires CUDA_HOME/NVCC")

    sources = [
        "src/fast_compressai/coders/cuda/ans_gpu.cpp",
        "src/fast_compressai/coders/cuda/warp_ans_kernel.cu",
        "src/fast_compressai/coders/cuda/warp_ans_kernel_v2.cu",
        "src/fast_compressai/coders/cuda/warp_index_check.cu",
    ]
    nvcc_flags = ["-O3"]
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not arch_list:
        raise RuntimeError(
            "Set TORCH_CUDA_ARCH_LIST explicitly when building the optional CUDA extension"
        )
    ext_modules = [
        CUDAExtension(
            name="fast_compressai.coders._warp_div",
            sources=sources,
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
