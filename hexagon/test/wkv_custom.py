wkv_c_impl_src = """
#include <torch/extension.h>
#include <torch/script.h>

std::tuple<torch::Tensor, torch::Tensor> wkv6(
    torch::Tensor k, torch::Tensor v, torch::Tensor r,
    torch::Tensor state2, torch::Tensor time_first,
    torch::Tensor time_decay) {
    k = torch::unsqueeze(k, -1);
    v = torch::unsqueeze(v, 1);
    r = torch::unsqueeze(r, 1);
    auto kv = torch::matmul(k, v);
    auto wkv = torch::matmul(r, (time_first * kv + state2));
    auto new_state2 = time_decay * state2 + kv;
    return std::make_tuple(wkv, new_state2);
}

std::tuple<torch::Tensor, torch::Tensor> wkv6_chunk(
    torch::Tensor k, torch::Tensor v, torch::Tensor r,
    torch::Tensor state2, torch::Tensor time_first,
    torch::Tensor time_decay) {
    // TODO
    auto num_head = state2.size(0);
    auto head_size = state2.size(2);
    return std::make_tuple(torch::zeros({32, num_head, 1, head_size}), state2);
}

TORCH_LIBRARY(rwkv, m) {
  m.def("wkv6", &wkv6);
  m.def("wkv6_chunk", &wkv6_chunk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
"""