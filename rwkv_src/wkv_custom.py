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
    auto seq_length = k.size(0) / num_head;
    return std::make_tuple(torch::zeros({seq_length, num_head, 1, head_size}), state2);
}

std::tuple<torch::Tensor, torch::Tensor> wkv7(
    torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor state2) {
    k = torch::unsqueeze(k, -1);
    a = torch::unsqueeze(a, -1);
    b = torch::unsqueeze(b, -2);
    v = torch::unsqueeze(v, -2);
    r = torch::unsqueeze(r, -2);
    auto kv = torch::matmul(k, v);
    auto ab = torch::matmul(a, b);
    auto new_state2 = w * state2 + kv + torch::matmul(ab, state2);
    auto x = torch::matmul(r, new_state2);
    return std::make_tuple(x, new_state2);
}

std::tuple<torch::Tensor, torch::Tensor> wkv7_chunk(
    torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor state2) {
    // TODO
    auto num_head = state2.size(0);
    auto head_size = state2.size(2);
    auto seq_length = k.size(0) / num_head;
    return std::make_tuple(torch::zeros({seq_length, num_head, 1, head_size}), state2);
}

TORCH_LIBRARY(rwkv, m) {
  m.def("wkv6", &wkv6);
  m.def("wkv6_chunk", &wkv6_chunk);
  m.def("wkv7", &wkv7);
  m.def("wkv7_chunk", &wkv7_chunk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
"""