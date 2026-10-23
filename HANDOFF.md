# Fast-CompressAI 开发交接文档

> 本文件用于新的 Claude Code 窗口快速恢复项目上下文。
>
> 当前日期：2026-10-22
>
> 当前仓库：`/hwj/project/fast-compressai`
>
> 项目来源论文目录：`/hwj/project/ppopp27`

---

## 1. 当前项目背景

用户此前完成了一篇 PPoPP 2027 论文：

> **CAIEC: Co-Accelerating Inference and Entropy Coding for Learning-Based Scientific Data Compression**

论文核心技术包括：

1. component-wise mixed-precision inference；
2. 面向 learned latent 的 GPU ANS entropy coding，即 GA-ANS；
3. 根据 inference 时间选择 coding granularity `P_ch`；
4. 使用 CUDA streams/events 对 inference 和 entropy coding 做 pipeline overlap；
5. 面向 scientific tensors 的 normalization、tiling 和数据适配。

论文收到三位审稿人的主要批评：

- GA-ANS 的 chunk/warp/sub-chunk 映射和内存布局不够可复现；
- `P_ch` 的选择方法没有明确说明；
- GA-ANS 内部优化缺少细粒度 incremental ablation；
- baseline 可能偏弱；
- profiling/configuration 成本没有量化；
- 大量缩写未定义，background 不够 self-contained；
- 方法整体被认为是已有技术的组合，创新性偏 incremental。

用户决定暂不 rebuttal，也不直接原样改投，而是重新建设一个更通用、更易用的系统和后续论文方向。

---

## 2. Fast-CompressAI 的总体定位

Fast-CompressAI 是一个**独立、可安装、面向用户的通用加速框架**，目标是让使用 CompressAI 风格接口编写的图像/科学数据压缩模型能够方便地获得加速。

它不是：

- CompressAI 的简单 fork；
- 只服务论文中固定模型的脚本；
- 强制依赖 TensorRT 的部署流程；
- 为所有模型预先手写完整 CUDA kernel 的系统；
- 只针对科研实验的临时仓库。

推荐定位：

> Fast-CompressAI is an extensible acceleration layer for CompressAI-style learned compression models. It preserves a usable FP32 reference path for new operators, while allowing users to register low-precision implementations that can participate in quality-aware policy selection.

中文表述：

> Fast-CompressAI 是面向 CompressAI 风格 learned compression 模型的可扩展加速层。新模型可以直接接入；未知算子默认使用 FP32 reference fallback；用户可以进一步提供自己的低精度实现，使其参与统一的 profiling、精度选择和端到端加速流程。

---

## 3. 与 CAIEC 的关系

Fast-CompressAI 继承 CAIEC 的三个核心研究方向，但进行系统级重构：

| CAIEC/PPoPP | Fast-CompressAI |
|---|---|
| 特定 scientific compression workflow | CompressAI 风格模型和用户自定义算子 |
| TensorRT/特定部署路径 | Python/PyTorch 入口，底层可插入 PyTorch/Triton/CUDA backend |
| component-wise mixed precision | 可扩展的 component/operator-aware `PrecisionPolicy` |
| GA-ANS 是 CAIEC 内部 coder | GA-ANS 是可替换的 `CoderBackend` |
| pipeline 是论文系统中的执行机制 | pipeline 是可替换的 `PipelineScheduler` |
| 固定模型与实验流程 | 新模型先运行，关键算子再逐步优化 |
| scientific-data adaptation 较突出 | scientific adaptation 作为 adapter/data layer，核心先服务 CompressAI |

不要把 Fast-CompressAI 描述为“把 CAIEC 从 TensorRT 改写成 Python”。真正的变化是：

```text
CAIEC 的算法机制
        ↓
独立的 adapter / operator registry / policy / backend / runtime
        ↓
可被用户使用、替换、升级和论文消融的框架
```

---

## 4. 核心用户体验和公共 API

面向 Python 用户的目标入口：

```python
import fast_compressai

fast_model = fast_compressai.accelerate(
    model,
    P=8,
    budget=0.1,
    device="cuda",
)

strings = fast_model.compress(x)
x_hat = fast_model.decompress(strings, shape)
```

公开 API 提供系统目标与约束。用户不应手工指定 module path 或 precision map。框架内部负责模型分析、候选能力检查、profiling、质量预算下的 precision selection、coder granularity/co-design、runtime 应用和报告。

参数语义：

- `P` 是 GA-ANS 的编码粒度参数 `P_ch`：每个独立 chunk 包含的连续 latent channel 数。它不是 precision。
- `budget` 是质量/RD budget，论文中以 `epsilon` 表示，并约束 acceleration-induced RD degradation。
- inference-time co-design constraint `T_ec(P_ch) <= T_inf(C*)` 是内部 selector 使用的测量约束，不是用户 `budget`。
- `P_ch` 合法范围、是否要求整除 channel count，以及 quality metric 的通用 schema 需要由后续 coder/model profile contract 明确；不能在 API 层臆定。
- 未来可增加 representative data/profile cache/pipeline options 等明确配置项；不要使用无校验的任意 `**kwargs`。

目标是尽量保留：

```python
fast_model.forward(x)
fast_model.compress(x)
fast_model.decompress(strings, shape)
fast_model.state_dict()
```

`compile()` 是当前兼容性/reference inspection 入口；`accelerate()` 是自动分析与配置入口。自动 profiling、budget-aware precision selection、GA-ANS、runtime transformation 尚未完成前，`accelerate()` 必须如实标记 reference-only，不声称已经加速。

---

## 5. 内部系统边界

```text
Public accelerate(model, P_ch, quality_budget, device, ...)
        ↓
Model adapter / supported operator inventory
        ↓
Operator registry and capability checks
        ↓
Profiling: latency, quality/RD, coder T_ec/CR, setup cost
        ↓
Budget-aware precision selection with FP32 reference fallback
        ↓
GA-ANS granularity selection (P_ch) and co-design
        ↓
Runtime application / sequential then overlapped scheduler
        ↓
Correctness, quality, measured end-to-end report
```

### 5.1 Public orchestration

负责校验 model、`P`/`P_ch`、quality budget、device 与未来 profile/runtime options；调用内部 model analysis、profiling、selection 和 runtime。内部 precision map 与 module paths 属于实现数据，不是普通用户必须提供的 API。

### 5.2 Model adapters and operator inventory

CompressAI adapter 识别 `forward/compress/decompress/update` 等公开路径；后续 model analysis 要自动枚举可优化组件并为其绑定内部 stable identity。未知算子保留其 FP32/reference 执行；候选未注册、设备/shape 不满足或低精度实现不可用时必须 fallback 并说明原因。

### 5.3 Operator registry and PrecisionPolicy

Registry 保存 reference 与可选低精度实现及 capability。PrecisionPolicy/selector 是内部可替换算法：读取 profiling 和 quality budget，输出组件/operator precision map；对完整配置做质量验证和必要的 precision recovery。用户-facing `accelerate()` 不要求用户提供 precision policy/map。显式 low-level registry API 可用于插件、测试和研究，不代表自动模型支持已完成。

### 5.4 Profiling and metrics

统一 profile schema 记录 model/component/operator/coder candidate、输入形状、device/software context、精度或 `P_ch`、latency/throughput/CR、quality/RD delta、重复统计、profile/configuration cost、provenance 与不可用原因。区分 measured、estimated、imported；选择算法只能把测量条件相容的数据相比较。

### 5.5 CoderBackend and granularity

`CoderBackend` 定义 session/encode/decode/finalize；CompressAI RANS adapter 是 correctness/reference 基础，不是 GA-ANS。GA-ANS 后续负责 chunk/warp/sub-chunk mapping、memory layout 等，并对 `P_ch` 形成 `T_ec`–CR frontier。`P_ch` 由内部 granularity/co-design controller 选取，不硬编码到模型 adapter。

### 5.6 PipelineScheduler and runtime

先 sequential correctness，再 staged non-overlap，再 streams/events overlap、buffer ownership/reuse 和 startup/steady-state/drain measurements。只有验证真实执行路径和测量结果后，才能报告 runtime precision applied、overlap 或 speedup。Co-design 目标必须考虑执行模式：sequential 与已验证 overlap 的端到端目标不同。

---

## 6. 仓库架构蓝图

现有代码按明确边界逐步演进，不需要一次性生成所有目录：

```text
src/fast_compressai/
├── api.py                     # accelerate facade; compile compatibility API
├── config.py                  # acceleration/runtime config
├── report.py                  # truthful compatibility/selection/runtime report
├── adapters/                  # CompressAI model contracts and analysis
├── operators/                 # reference/low-precision candidates and registry
├── policies/                  # precision, granularity, pipeline, co-design policies
├── algorithms/                # precision recovery, P_ch selection, co-design
├── profiling/                 # profile schemas and collection
├── metrics/                   # quality/RD, throughput, latency
├── coders/                    # CoderBackend, reference coder, GA-ANS
├── backends/                  # PyTorch/Triton/CUDA implementations
└── runtime/                   # dispatch, model transform, FastModel, scheduler
```

仓库原则：

- 主仓库是可安装、面向用户的产品，不是论文临时脚本。
- `research/`/`benchmarks/` 保存实验配置、消融、raw/processed results 和 figure workflow，不污染主 API。
- algorithms 存研究算法，policies 存可替换策略，backends/coders 存具体实现。
- `PrecisionPolicy`、`CoderBackend`、`PipelineScheduler` 可独立替换并消融。
- 保留 reference path；report 区分 selected、runtime-applied、quality-validated 和 measured speedup。
- 不把 CAIEC 旧论文数字当 Fast-CompressAI 结果；所有 speedup/quality/profile cost 重新测量。

---

## 7. 当前真实进度（2026-10-22）

### 已有实现 / 已验证边界

- 包含公开 `fast_compressai.accelerate(model, P=..., budget=..., ...)` 入口及 FactorizedPrior runtime；`P` 是 GA-ANS 的 `P_ch`，默认 coder=`warp_div`，显式 `coder="rans"` 使用原模型 RANS 路径。CPU/不支持模型不会静默回退到 warp_div。
- WarpANS V2 fast-division encode + V1-compatible decode 已移植到自有 CUDA namespace。WarpDiv 输出 device-local `WarpDivPayload`（packed tensor/metadata），不是可持久化的 bytes bitstream。此前在 Torch 2.8.0+cu126/CUDA 12.6/H100 sm90 编译并通过基础 GPU round-trip；其他环境和广泛 shape/layout coverage 未验证。
- Balle operator catalog/inventory 识别逐实例 Conv2d、ConvTranspose2d、GDN/IGDN、EntropyBottleneck；提供 FP32、FP16、BF16 candidate factory。EntropyBottleneck 的 CDF/index/symbol/coder 状态 precision-locked，不进入普通混精选择。此前 H100 上 Balle BF16 candidate 测试通过；近期修改后需重跑全套测试。
- `PrecisionConfig` 有版本化 JSON、逐实例 candidate assignment、fingerprints、probe/recovery records 和原子保存/加载的实现。目标语义是：传 config 只校验并应用，不重选；未传 config 时用调用方传入的校准数据选择并按 model+dataset+UTC timestamp 自动保存。
- Kodak loader 在 `examples/`/profiling helper 层，由调用方加载后传给 `accelerate()`；文件语义固定为 NCHW raw `[0,255]`，模型输入 `/255`，重建输出 `*255`，PSNR `data_range=255`。RD Cost 使用 `log10(bpp/bpp_ref) + (PSNR_ref - PSNR)`。
- `examples/balle_example.py` 已支持本地 Balle checkpoint（不联网下载）、Kodak 校准、precision config 参数及 RANS/WarpDiv 压缩解压计时。example 尚未完成真实 Kodak CAIEC 流程验证。
- CAIEC selector 有 all-FP16（EntropyBottleneck 除外）reference、逐实例 FP8 probe、初始分配及 recovery 流程雏形。已写过 mock orchestrator 单测；它不等于真实 GPU/Kodak 选择通过。对于缺少 FP8 candidate 的实例应写 unavailable 状态，不产生伪造 RD/throughput 数字。
- CPU 单测曾在较早代码状态下通过 79 tests；此后有代码变更，当前整体测试状态未知。H100 上 Triton FP8 测试最近失败，见下方 blocker。

### 当前明确 blocker / 未完成

- **Triton FP8 Conv2d 数值尚未达标。** `tests/gpu/test_triton_fp8.py::test_non_aligned_conv2d` 最近结果 mean absolute error `0.341016...`，断言阈值 `<0.2`。当前实现由 Triton 生成 scaled/clamped FP32 tensor，再由 Torch cast E4M3FN，具体量化/scale 或计算语义误差尚未定位。不能将此候选宣称为已验证，也不能据此声称 Kodak selector 跑通。
- Catalog 当前 Conv2d FP8 指向 Triton；GDN 仍注册 PyTorch scaled-mm FP8 candidate，与“先做 Triton FP8”的当前实施方向不一致。ConvTranspose2d、GDN/IGDN 若暂时没有经验证的 Triton FP8 实现，应由 CAIEC 为每个实例记录 FP8 unavailable，并保守使用可用 FP16/FP32 candidate；不要借用别的精度/backend 的数据冒充 FP8 sensitivity。
- 必须对每个 eligible Balle operator instance 做 one-at-a-time FP16→FP8 probe，或明确记录 unavailable；reference 以 EntropyBottleneck 锁定、其余 eligible 算子 all-FP16 为基线。每个可用 probe 的完整模型 RD 和 latency 均需实测；不可把各 probe 的 RD 相加替代 full-config Kodak RD。
- Kodak 上完整 all-FP16 reference、全部逐实例 probe、CAIEC 分类、初始混精配置、逐步 recovery/full-config 重测、预算 infeasible 报告、自动保存 config 尚未端到端验证。
- config load-only 路径尚需验证严格兼容性（model/checkpoint、dataset、catalog/inventory、coder、`P_ch` 等）并证明不会调用 profile/selector/recovery；不传 config 的路径必须验证默认命名和保存后的 round-trip。
- THR Gain 的延迟范围应采用论文定义的 inference/operator latency，不应混入 coder 的 compress/decompress 延迟。对所有测试及文档只报告实际测量结果。
- 最近代码之后需重跑 CPU unittest、GPU candidate 测试及完整 example；仓库当前环境未提供 git 元数据（`git status` 在项目目录报不是 git repository），不能据此汇报 diff 状态。

---

## 8. 后续开发顺序

1. **Public API/config/profile contract**：固定 `accelerate(model, P, budget, ...)`、P_ch 与 quality budget 的不同语义、版本化 profile schema 和 truthful report。
2. **Model analysis/operator inventory**：自动识别支持的组件与候选；内部稳定 path 不暴露为用户必填参数。
3. **Profiling and quality metrics**：测量 reference 与候选精度的性能和 RD impact，报告 setup/profile cost，保存可重用 profile。
4. **Budget-aware precision selection**：基于能力和 profile 生成内部 precision map；无低精度候选或质量不满足时使用 FP32；验证完整配置并实施 recovery。
5. **Coder granularity profiling**：基于已集成的 warp_div backend 与 device-local payload contract，扩展 latent shape/CDF/index/layout 测试并 sweep P_ch，形成 T_ec–CR frontier。
6. **GA-ANS kernel hardening**：现已移植 WarpANS V2 fast-division encode + V1-compatible decode；继续做更全面的 CUDA correctness、跨 batch/device、bypass、header/payload corruption 与资源边界验证；只保留 RANS 与 warp_div 两种 coder。
7. **Co-design**：满足 quality budget 后，结合 T_inf(C) 与 coder frontier 选择 P_ch；目标依据实际 execution mode 定义，不用未经验证的 max(T_inf,T_ec) 代替实测。
8. **Runtime and scheduler**：先使内部 precision map 真正应用到执行路径并保持 CompressAI API/state/device 正确；再做 sequential、staged 和 stream/event overlap。
9. **End-to-end evaluation**：真实模型验证 RD/quality、bitstream round-trip、latency/throughput、fallback coverage、profile/configuration overhead。

当前仓库已超出最初的 reference-only facade 阶段：WarpDiv FactorizedPrior runtime、Balle candidate catalog、CAIEC selector/config 和 Kodak loader 均已有实现片段；但 Kodak CAIEC 端到端仍未完成，且 Triton FP8 Conv2d 数值测试失败。切勿沿用早期“完全没有 GA-ANS/自动选择”的描述，也不可把代码雏形或 mock 测试描述为真实端到端能力。下一步先解决 FP8 数值 blocker，再验证逐实例 probe 覆盖、Kodak recovery/config round-trip 和完整测试。

---

## 9. 第一阶段架构验收标准

- `fast_compressai.accelerate(model, P=..., budget=...)` 是用户入口，P 映射 P_ch，budget 表示 quality/RD constraint。
- CompressAI 风格模型的 forward/compress/decompress/state_dict/device 兼容边界明确。
- 用户 API 不要求 module path 或内部 precision map。
- FP32/reference path 始终可用；未注册/不满足 capability 的低精度候选内部 fallback 并可解释。
- profile schema 能表达精度与 coder 参数的 measured/estimated 数据及上下文。
- 当前未实现 profiling/selector/GA-ANS/runtime 时必须明确标记 not implemented/not applied，不报告 speedup。
- 单元测试覆盖配置、序列化、adapter、fallback、bitstream 和 truthful status。

---

---

## 10. 开发注意事项

1. **先确认目录中的实际内容，再编辑文件。** 本交接文件只记录当前设计，不代表目录一定为空。
2. **实现前优先进入 plan mode**，当任务涉及多个模块或架构变化时，先给出计划再修改。
3. **不要把 CAIEC 论文中的数字直接当成 Fast-CompressAI 结果。** 论文中的 speedup、吞吐率和质量结果属于旧系统，后续必须重新测量。
4. **不要默认 TensorRT 是必需依赖。** TensorRT 可以作为可选 backend 或 baseline，但核心 Python API 不应依赖它。
5. **不要一开始实现所有算子。** 先做 registry、fallback 和接口，再根据 profiling 选择关键算子。
6. **不要把每个图节点强行独立搜索。** 第一版可以从 component-wise 或 operator-group policy 开始，防止搜索空间爆炸。
7. **任何 pipeline overlap claim 都必须有实测证据。** 理论 `max(T_inf,T_ec)` 不能代替真实 benchmark。
8. **reference backend 必须始终保留。** 它用于 correctness、回归测试和论文 baseline。
9. **核心产品 API 与论文实验逻辑分离。** 论文实验放在 `research/`/`benchmarks/`，不要污染 `src/fast_compressai/`。
10. **优先建立可解释的报告。** 用户需要知道 quality budget、`P_ch`、fallback、所选 precision/backend、profile 来源/成本、runtime 是否应用以及真实测量结果。
11. **公开 API 隐藏内部 precision map 和 module path。** 用户提供模型和目标约束；内部自动分析、profiling、选择与 fallback。
12. **不要混淆参数。** `P`/`P_ch` 是 GA-ANS chunk granularity；quality `budget` 是 RD 约束；`T_ec <= T_inf` 是内部 co-design 条件。
13. **先兑现实际能力，再报告选择。** 选择到候选实现不等于替换已应用；报告中的 accelerated/speedup 必须对应真实运行路径和测量。
14. **profile 结果必须带上下文与来源。** model/operator、device、输入 shape、precision/coder 参数、quality 与测量条件不兼容时不能直接比较。

---

## 11. 新窗口恢复方式

新窗口应先阅读本文件、README 和当前实际代码。当前工作目标是完成 Balle FactorizedPrior 的 Kodak CAIEC 算子级混合精度流程，而非重建基础 API 或重新移植 GA-ANS。优先顺序：

1. 首先检查 `src/fast_compressai/backends/triton/fp8_conv2d.py` 与 `tests/gpu/test_triton_fp8.py`，定位 E4M3FN scale/量化/`_scaled_mm` 计算造成的误差；当前 H100 最近测试 mean absolute error 约 `0.341`，阈值 `<0.2`，测试失败。修复后单独重跑此 GPU 测试，未通过前不要把 candidate 注册为可用于自动选择的已验证实现。
2. 核验 `operators/catalog.py`：当前 Triton FP8 仅 Conv2d，GDN 尚有 PyTorch scaled-mm FP8 registration。按“先 Triton”的方向调整候选可用性；无实现的 eligible instance 必须有 unavailable probe record。
3. 检查 `policies/caiec_selection.py`、`profiling/rd.py`、`api.py`：one-at-a-time probe 要覆盖每个 eligible instance；EntropyBottleneck 锁定且不进入 FP16/FP8混精 eligible 集；probe latency 与 full-model Kodak RD 分开记录；recovery 的每一步都应完整重测。
4. 验证调用方加载 Kodak 后传 `CalibrationData` 给 `accelerate()`；config path 只校验和应用、不重选；无 config 时选择并保存 model+dataset+UTC timestamp 配置。更新/运行 `examples/balle_example.py` 完成端到端验证。
5. 重新跑 CPU unittest、H100 candidate tests 和 example。明确记录未运行项/失败输出，不把 mock tests 当成 GPU/Kodak 验收。

API/设计约束：`P` 是 GA-ANS `P_ch`；默认 coder 为 `warp_div`，显式 RANS 保持可用；EntropyBottleneck 离散 CDF/index/symbol/coder state 锁定；Kodak raw NCHW `[0,255]` 输入模型前 `/255`，重建后 `*255` 再用 `data_range=255` 计算 PSNR；RD Cost 为 `log10(bpp/bpp_ref) + (PSNR_ref - PSNR)`。
