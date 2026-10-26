# Fast-CompressAI

Fast-CompressAI is an extensible acceleration layer for CompressAI-style learned compression models. Its target workflow is `accelerate(model, P=..., budget=...)`: the library profiles candidate implementations, chooses internal operator precision under a quality constraint, selects coding granularity, and applies a runtime plan while retaining an FP32 reference fallback.

## Current Status

The repository provides an installable PyTorch package, CompressAI compatibility inspection, an explicit operator registry, reporting-only precision policies, a reference CompressAI RANS adapter, and WarpANS V2 `warp_div`. `accelerate()` can explicitly route Balle `FactorizedPrior` compression/decompression through the local CUDA GA-ANS backend; a separate CPU `sample=` experiment profiles and conditionally applies BF16 Conv2d modules. Automatic per-operator precision selection, joint quality-budget/co-design search, pipeline overlap, and general CompressAI model support are not yet implemented. Neither path implies an end-to-end speedup without a matching benchmark.

## Balle Model Example

After installing CompressAI and building the optional CUDA extension, run the included Balle model example:

```bash
FAST_COMPRESSAI_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
python examples/balle_example.py --image ./image.png --quality 1 --p-ch 4
```

For Kodak calibration, omit `--p-ch` to automatically select the smallest P whose measured encoding time is close to the final mixed-precision model's inference time. The selector uses a relative-time tolerance and detects a large-P timing plateau as measurement variation; it does not treat small noisy changes in the plateau as algorithmic improvements:

```bash
python examples/balle_example.py \
    --kodak /path/to/Kodak_24x3x512x768.f32 \
    --weights /path/to/bmshj2018-factorized-1.pth \
    --precision-budget 0.1
```

Passing `--p-ch N` keeps P fixed and disables the automatic P sweep.

The example loads a local `bmshj2018-factorized` checkpoint and pads the RGB image to the model's downsampling factor. It uses the default `warp_div` coder and prints a compact reference-versus-accelerated comparison containing:

- compression and decompression throughput in decimal **GB/s**;
- bpp and PSNR for both paths;
- compression and decompression speedup.

For example, the command line prints a small summary like:

```json
{
  "reference": {
    "compress_throughput_gb_s": 0.42,
    "decompress_throughput_gb_s": 0.37,
    "bpp": 0.029,
    "psnr_db": 31.96
  },
  "accelerated": {
    "compress_throughput_gb_s": 4.10,
    "decompress_throughput_gb_s": 3.75,
    "bpp": 0.031,
    "psnr_db": 31.96
  },
  "compress_speedup": 9.65,
  "decompress_speedup": 10.10
}
```

The complete benchmark report, including timings, payload sizes, dataset metadata, and the acceleration report, is saved under `examples/reports/`. When Kodak precision selection is enabled, its selection, probe-record, and recovery JSON files are saved under `examples/precision_configs/`; pass `--precision-config` to reuse the final selection file. The returned packed payload is device-local, so this example does not save it as a portable bitstream.

## Install

```bash
python -m pip install -e .
```

CompressAI integration is optional:

```bash
python -m pip install -e '.[compressai]'
```

## Public API Shape

```python
import fast_compressai

model = MyCompressAIModel(...)
fast_model = fast_compressai.accelerate(
    model,
    P=8,
    device="cuda",
)

strings = fast_model.compress(x)
x_hat = fast_model.decompress(strings, shape)
print(fast_model.acceleration_report.to_dict())
```

`P` maps to GA-ANS `p_ch`: the number of latent channels per independently coded chunk. `budget` is currently interpreted as the maximum reconstruction `psnr_drop_db` in dB only for the CPU forward profiling slice (the paper uses `epsilon` for a broader RD degradation constraint). These are separate from the internal inference-time co-design constraint comparing entropy-coding time with inference time. `P` is applied to the GPU coder only when explicitly selecting `coder="warp_div"`; it is otherwise recorded but not applied. The low-level `WarpDivBackend` accepts explicit CUDA symbols/CDF/index/offset tensors and `p_ch`.

GA-ANS is the default coder. To explicitly retain the original RANS path, pass `coder="rans"`. To run the smallest Balle model through the default GA-ANS (requires the opt-in CUDA build and an initialized entropy table):

```python
import torch
from compressai.models import FactorizedPrior
import fast_compressai

model = FactorizedPrior(N=8, M=12).eval()
model.update(force=True)
x = torch.rand(1, 3, 64, 64, device="cuda")
fast_model = fast_compressai.accelerate(
    model, P=4, device="cuda"
)
encoded = fast_model.compress(x)
x_hat = fast_model.decompress(encoded["strings"], encoded["shape"])["x_hat"]
```

This path deep-copies the original model to CUDA and retains FP32 transforms. `encoded["strings"]` contains a `WarpDivPayload`, not CPU `bytes`. It must remain on the CUDA device with matching CDF tables and cannot yet be saved as a persistent CompressAI bitstream. The `budget` parameter is not used for coder-only selection or validated as an end-to-end RD constraint in this path. Other CompressAI models and `sample=`-driven BF16 selection with the default coder are rejected rather than silently falling back. Use `coder="rans"` for the original model RANS path.

The library will choose operator precision internally based on registered candidates, capability checks, profiling, and the quality budget. Users should not need to supply module paths or a precision map. When `sample=` is provided together with `coder="rans"`, the current CPU profiling slice measures a BF16 Conv2d candidate for supported models, applies it only when the forward PSNR-drop budget is met and the candidate is measured faster, and otherwise retains FP32. The selected operator candidates are intended to participate in the returned model's `forward`, `compress`, and `decompress` transforms. The current Balle operator catalog and factory layer can construct per-instance candidates for GPU validation; automatic precision assignment is still experimental and not enabled by default. The default `coder="warp_div"` path instead keeps FP32 transforms and applies GA-ANS to the supported FactorizedPrior entropy path. Automatic precision-map selection, automatic GA-ANS `P_ch` choice, and pipeline scheduling are not implemented. The legacy `compile()`/`optimize()` API remains available for compatibility inspection and explicit low-level experiments.

## Architecture

The intended internal flow is:

```text
accelerate(model, P, budget, device, ...)
    -> model adapter and operator inventory
    -> operator/coder profiling and quality measurements
    -> budget-aware precision selection with FP32 fallback
    -> GA-ANS p_ch selection and inference/coding co-design
    -> runtime application and sequential/overlapped scheduling
    -> correctness, quality, and measured performance report
```

`PrecisionPolicy`, `CoderBackend`, and `PipelineScheduler` are independent replaceable boundaries. The current precision policy is reporting-only. FP8 is currently an explicit experimental Conv2d candidate on H100-class GPUs using PyTorch's scaled-matrix-multiply primitive; it is not selected by default and is not available for ConvTranspose2d/GDN/IGDN or entropy coding. The supported coder catalog is limited to original RANS (`compressai-rans`) and WarpANS V2 `warp_div`. `accelerate()` defaults to `warp_div`; use `coder="rans"` to retain the model's original RANS path. The original RANS adapter imports optional CompressAI lazily. `warp_div` is an opt-in CUDA extension; its device-local payload is not the CPU `Bitstream` byte format. `warp_div` is V2 fast-division encoding with V1-compatible decode/packing, not V3/`warp_smem`. Its extension requires an explicit CUDA build; CPU-only install/import and explicit RANS remain available without it. The default GA-ANS request raises a clear error for unsupported models or CPU devices; it never silently switches coder formats.

To opt into the CUDA extension build in a CUDA-enabled development environment:

```bash
FAST_COMPRESSAI_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
```

Use the target deployment architecture in `TORCH_CUDA_ARCH_LIST`. The extension was built and smoke-tested on Torch 2.8.0+cu126, CUDA 12.6, H100 sm90. Other toolchains/architectures are not yet validated. See `NOTICE` for source provenance and migration authorization record.

## Profile Schema

Versioned `ProfileRecord` and `ProfileSet` objects define JSON-safe inputs for future operator-precision and coder-granularity selectors. Records carry model/operator or coder candidate identity, context, measurements, provenance (`measured`, `estimated`, or `imported`), and unavailable reasons. The schema is in place; automatic collection and profile-driven selection are future stages.
