// Adapter helper extracted from CompressAI-Science's authorized WarpANS source.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void check_idx_is_channel_kernel(
    const int32_t* __restrict__ indexes_bxn, int B, int N, int C, int HW,
    int32_t* __restrict__ flag) {
  int channel = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= C || *flag == 0) return;
  int base = channel * HW;
  if (base >= N) { atomicExch(flag, 0); return; }
  for (int batch = 0; batch < B; ++batch) {
    const int32_t* indexes = indexes_bxn + (int64_t)batch * N;
    int32_t value = indexes[base];
    if (value != channel) { atomicExch(flag, 0); return; }
    for (int pos = base + 1; pos < base + HW && pos < N; ++pos) {
      if (indexes[pos] != value) { atomicExch(flag, 0); return; }
    }
  }
}
