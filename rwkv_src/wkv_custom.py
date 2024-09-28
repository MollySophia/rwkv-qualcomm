wkv_c_impl_src = """
#include <torch/script.h>
#include <tuple>

static std::tuple<torch::Tensor, torch::Tensor> custom_wkv(
    torch::Tensor k, torch::Tensor v, torch::Tensor r,
    torch::Tensor state2, torch::Tensor time_first,
    torch::Tensor time_decay) {
    auto kv = torch::matmul(k, v);
    auto wkv = torch::matmul(r, (time_first * kv + state2));
    auto new_state2 = time_decay * state2 + kv;
    return std::make_tuple(wkv, new_state2);
}

TORCH_LIBRARY(rwkv, m) {
  m.def("custom_wkv", &custom_wkv);
}
"""