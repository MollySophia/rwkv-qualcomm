wkv_c_impl_src = """
#include <torch/extension.h>
#include <torch/script.h>

std::tuple<torch::Tensor, torch::Tensor> wkv6(
    torch::Tensor k, torch::Tensor v, torch::Tensor r,
    torch::Tensor state2, torch::Tensor time_first,
    torch::Tensor time_decay) {
    state2 = state2.squeeze(0);
    auto num_head = state2.size(0);
    auto head_size = state2.size(1);
    int seq_length = k.size(0) / num_head;

    k = k.reshape({seq_length, num_head, head_size, 1});
    v = v.reshape({seq_length, num_head, 1, head_size});
    r = r.reshape({seq_length, num_head, 1, head_size});
    time_first = time_first.reshape({num_head, head_size, 1});
    time_decay = time_decay.reshape({seq_length, num_head, head_size, 1});
    auto kv = torch::matmul(k, v);
    std::vector<torch::Tensor> wkv;
    for (int i = 0; i < seq_length; i++) {
        wkv.push_back(torch::matmul(r[i], (time_first * kv[i] + state2)));
        state2 = time_decay[i] * state2 + kv[i];
    }
    auto wkv_tensor = torch::stack(wkv, 0).reshape({seq_length * num_head, head_size});

    return std::make_tuple(wkv_tensor, state2);
}

std::tuple<torch::Tensor, torch::Tensor> wkv7(
    torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor state2) {
    state2 = state2.squeeze(0);
    auto num_head = state2.size(0);
    auto head_size = state2.size(1);
    int seq_length = k.size(0) / num_head;

    w = w.reshape({seq_length, num_head, head_size, 1});
    k = k.reshape({seq_length, num_head, head_size, 1});
    v = v.reshape({seq_length, num_head, 1, head_size});
    r = r.reshape({seq_length, num_head, 1, head_size});
    b = b.reshape({seq_length, num_head, head_size, 1});
    a = a.reshape({seq_length, num_head, 1, head_size});

    auto kv = torch::matmul(k, v);
    auto ab = torch::matmul(b, a);
    std::vector<torch::Tensor> x;
    for (int i = 0; i < seq_length; i++) {
        state2 = w[i] * state2 + kv[i] + torch::matmul(ab[i], state2);
        x.push_back(torch::matmul(r[i], state2));
    }
    auto x_tensor = torch::stack(x, 0).reshape({seq_length * num_head, head_size});

    return std::make_tuple(x_tensor, state2);
}

TORCH_LIBRARY(rwkv, m) {
  m.def("wkv6", &wkv6);
  m.def("wkv7", &wkv7);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
"""