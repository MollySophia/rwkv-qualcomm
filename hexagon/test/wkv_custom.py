wkv_c_impl_src = """
#include <torch/extension.h>
#include <torch/script.h>

std::tuple<torch::Tensor, torch::Tensor> wkv6(
    torch::Tensor k, torch::Tensor v, torch::Tensor r,
    torch::Tensor state2, torch::Tensor time_first,
    torch::Tensor time_decay) {
    auto num_head = state2.size(1);
    auto head_size = state2.size(2);
    auto seq_length = k.size(0);

    k = torch::unsqueeze(k, -1);
    v = torch::unsqueeze(v, -2);
    r = torch::unsqueeze(r, -2);
    auto kv = torch::matmul(k, v);
    std::vector<torch::Tensor> wkv;
    for (int i = 0; i < seq_length; i++) {
        wkv.push_back(torch::matmul(r[i], (time_first * kv[i] + state2)));
        state2 = time_decay[i] * state2 + kv[i];
    }
    auto wkv_tensor = torch::stack(wkv, 0);

    return std::make_tuple(wkv_tensor, state2);
}

std::tuple<torch::Tensor, torch::Tensor> wkv7(
    torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor state2) {
    auto num_head = state2.size(1);
    auto head_size = state2.size(2);
    auto seq_length = k.size(0);

    k = torch::unsqueeze(k, -1);
    v = torch::unsqueeze(v, -2);
    a = torch::unsqueeze(a, -1);
    b = torch::unsqueeze(b, -2);
    r = torch::unsqueeze(r, -2);

    auto kv = torch::matmul(k, v);
    auto ab = torch::matmul(a, b);
    std::vector<torch::Tensor> x;
    for (int i = 0; i < seq_length; i++) {
        state2 = w[i] * state2 + kv[i] + torch::matmul(ab[i], state2);
        x.push_back(torch::matmul(r[i], state2));
    }
    auto x_tensor = torch::stack(x, 0);

    return std::make_tuple(x_tensor, state2);
}

TORCH_LIBRARY(rwkv, m) {
  m.def("wkv6", &wkv6);
  m.def("wkv7", &wkv7);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
"""