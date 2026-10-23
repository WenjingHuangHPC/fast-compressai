# Optional WarpANS V2 CUDA Extension

The extension is deliberately not part of the default CPU package install. In an environment with a CUDA-enabled PyTorch, CUDA toolkit, NVCC, and CUB headers, build the current package in-place with:

```bash
FAST_COMPRESSAI_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
```

Set `TORCH_CUDA_ARCH_LIST` to the architectures intended for the target deployment. This build entry compiles only the V2 fast-division encode path and the V1-compatible decode/packing helpers required by its format.

The Python adapter returns device-local packed tensors plus metadata. The payload is not the CPU `Bitstream` byte format and is not currently a persistent cross-device/cross-version file format. Runtime support requires a CUDA device and the built `fast_compressai.coders._warp_div` extension. The build and tests must be repeated for every supported PyTorch/CUDA/compiler combination.

The migrated source subset and original-source notices are documented in the repository `NOTICE` file.
