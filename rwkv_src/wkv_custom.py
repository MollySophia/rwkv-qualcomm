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
    int seq_length = k.size(0);

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
    auto wkv_tensor = torch::stack(wkv, 0).reshape({seq_length, num_head, 1, head_size});

    return std::make_tuple(wkv_tensor, state2);
}

torch::Tensor wkv7(
    torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor state2) {
    state2 = state2.squeeze(0);
    auto num_head = state2.size(0);
    auto head_size = state2.size(1);
    int seq_length = k.size(0);

    w = w.reshape({seq_length, num_head, 1, head_size});
    k = k.reshape({seq_length, num_head, 1, head_size});
    v = v.reshape({seq_length, num_head, head_size, 1});
    b = b.reshape({seq_length, num_head, 1, head_size});
    a = a.reshape({seq_length, num_head, head_size, 1});
    r = r.reshape({seq_length, num_head, head_size, 1});

    auto kv = torch::matmul(v, k);
    auto ab = torch::matmul(a, b);
    auto state2_out = torch::zeros({num_head, head_size, head_size}, kv.options());
    auto wkv = torch::zeros({seq_length, num_head, head_size}, kv.options());
    for (int i = 0; i < seq_length; i++) {
        if (i == 0) {
            state2_out = w[i] * state2 + kv[i] + torch::matmul(state2, ab[i]);
        } else {
            state2_out = w[i] * state2_out + kv[i] + torch::matmul(state2_out, ab[i]);
        }
        wkv[i] = torch::matmul(state2_out, r[i]).reshape({num_head, head_size});
    }
    wkv = wkv.permute({1, 0, 2}); // (seq_length, num_head, head_size) -> (num_head, seq_length, head_size)
    auto output = torch::cat({wkv, state2_out}, 1); // (num_head, seq_length + head_size, head_size)
    return output;
}

torch::Tensor wkv7_output_x(torch::Tensor in) {
    auto num_head = in.size(0);
    auto head_size = in.size(2);
    int seq_length = in.size(1) - head_size;

    auto x = torch::zeros({seq_length, num_head, 1, head_size}, in.options());
    for (int i = 0; i < seq_length; i++) {
        x[i] = in.index({torch::indexing::Slice(), i, torch::indexing::Slice()}).reshape({num_head, 1, head_size});
    }
    return x;
}

torch::Tensor wkv7_output_state(torch::Tensor in) {
    auto num_head = in.size(0);
    auto head_size = in.size(2);
    int seq_length = in.size(1) - head_size;
    auto state = in.index({torch::indexing::Slice(), torch::indexing::Slice(seq_length, in.size(1)), torch::indexing::Slice()}).reshape({num_head, head_size, head_size});
    return state;
}

TORCH_LIBRARY(rwkv, m) {
  m.def("wkv6", &wkv6);
  m.def("wkv7", &wkv7);
  m.def("wkv7_output_x", &wkv7_output_x);
  m.def("wkv7_output_state", &wkv7_output_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
"""