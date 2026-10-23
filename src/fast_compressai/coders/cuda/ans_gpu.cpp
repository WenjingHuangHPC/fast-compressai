// Adapted from CompressAI-Science/compressai/cpp_exts/rans_gpu/ans_gpu.cpp.
// Source migration authorized by the project user; see NOTICE for provenance.
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> encode_with_indexes_warp_v2_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P_in);

torch::Tensor decode_with_indexes_warp_cuda(
    torch::Tensor packed_u8,
    torch::Tensor max_rounds_u32,
    torch::Tensor header_bytes_cpu,
    torch::Tensor chunk_len_cpu,
    torch::Tensor P_cpu,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("encode_warp_div", &encode_with_indexes_warp_v2_cuda,
             "WarpANS V2 fast-division encode");
  module.def("decode_warp_div", &decode_with_indexes_warp_cuda,
             "WarpANS V1-compatible decode for V2 payload");
}
