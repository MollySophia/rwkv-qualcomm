import torch
import torch.utils.cpp_extension
import numpy as np
import os, sys
import subprocess

from wkv_custom import wkv_c_impl_src
module = torch.utils.cpp_extension.load_inline(
        name='extension', cpp_sources=[wkv_c_impl_src])

n_head = 32
head_size = 64

cmd_stdout = subprocess.DEVNULL
cmd_stderr = subprocess.DEVNULL
# cmd_stdout = None
# cmd_stderr = None

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]

class qnn_graph_v6(torch.nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.wkv_func = torch.ops.rwkv.wkv6
        self.tf = torch.rand(n_head, head_size)

    def forward(self, k, v, r, state, td):
        return self.wkv_func(k, v, r, state, self.tf, td)

class qnn_graph_v7(torch.nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.wkv_func = torch.ops.rwkv.wkv7

    def forward(self, r, w, k, v, a, b, state):
        return self.wkv_func(r, w, k, v, a, b, state)

def onnx_custom_wkv6(g, k, v, r, state2, time_first, time_decay):
    n_head = state2.type().sizes()[0]
    head_size = state2.type().sizes()[1]
    out1, out2 = g.op("rwkv::wkv6", k, v, r, state2, time_first, time_decay, outputs=2)
    return out1.setType(k.type().with_dtype(torch.float32).with_sizes([k.type().sizes()[0], head_size])),\
        out2.setType(k.type().with_dtype(torch.float32).with_sizes([n_head, head_size, head_size]))

def onnx_custom_wkv7(g, r, w, k, v, a, b, state):
    n_head = state.type().sizes()[0]
    head_size = state.type().sizes()[1]
    out1, out2 = g.op("rwkv::wkv7", r, w, k, v, a, b, state, outputs=2)
    return out1.setType(k.type().with_dtype(torch.float32).with_sizes([k.type().sizes()[0], head_size])),\
        out2.setType(k.type().with_dtype(torch.float32).with_sizes([n_head, head_size, head_size]))

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic("rwkv::wkv6", onnx_custom_wkv6, 9)
register_custom_op_symbolic("rwkv::wkv7", onnx_custom_wkv7, 9)

def test_wkv6_cpu(n_head, head_size, seq_length):
    print(f"!!! Testing wkv6 cpu length = {seq_length}")
    gen_graph = qnn_graph_v6(n_head, head_size)
    k = torch.rand(seq_length * n_head, head_size)
    v = torch.rand(seq_length * n_head, head_size)
    r = torch.rand(seq_length * n_head, head_size)
    td = torch.rand(seq_length * n_head, head_size)
    state = torch.rand(n_head, head_size, head_size)
    inputs = (k, v, r, state, td)
    print("!!! converting test graph")
    torch.onnx.export(gen_graph, inputs, "test_wkv.onnx", output_names=["output", "state"], opset_version=17)
    subprocess.call("qnn-onnx-converter -i test_wkv.onnx --input_layout state.1 NONTRIVIAL --op_package_config ../RwkvWkvOpPackageCPU.xml", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)

    os.path.exists("test_data_wkv") or os.mkdir("test_data_wkv")
    for i in range(len(inputs)):
        inputs[i].numpy().tofile(f"test_data_wkv/input{i}.bin")
    input_list_lines = [" ".join([f"test_data_wkv/input{i}.bin" for i in range(5)])]
    with open("input_list_wkv.txt", "w") as f:
        f.writelines(input_list_lines)
    print("!!! generating qnn model lib")
    subprocess.call("qnn-model-lib-generator -c test_wkv.cpp -b test_wkv.bin -t x86_64-linux-clang", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)
    print("!!! executing qnn-net-run")
    subprocess.call(f"qnn-net-run --input_list input_list_wkv.txt --model lib/x86_64-linux-clang/libtest_wkv.so --backend {qnn_sdk_root}/lib/x86_64-linux-clang/libQnnCpu.so --op_packages ../CPU/RwkvWkvOpPackage/libs/x86_64-linux-clang/libRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider", stdout=None, stderr=None, shell=True)
    qnn_output = torch.from_numpy(np.fromfile("output/Result_0/output.raw", dtype=np.float32)).reshape(seq_length * n_head, head_size)
    qnn_state = torch.from_numpy(np.fromfile("output/Result_0/state.raw", dtype=np.float32)).reshape(n_head, head_size, head_size)
    torch_output, torch_state = gen_graph(*inputs)
    if not torch.allclose(qnn_output, torch_output, rtol=1e-5, atol=1e-5) or not torch.allclose(qnn_state, torch_state, rtol=1e-5, atol=1e-5):
        print(f"!!! wkv6 cpu length={seq_length} output mismatch!!!")
        print((qnn_output - torch_output).sum())
        print((qnn_state - torch_state).sum())
    else:
        print(f"!!! wkv6 cpu length={seq_length} output passed")

def test_wkv6_htp(n_head, head_size, seq_length):
    print(f"!!! Testing wkv6 htp length = {seq_length}")
    gen_graph = qnn_graph_v6(n_head, head_size)
    k = torch.rand(seq_length * n_head, head_size)
    v = torch.rand(seq_length * n_head, head_size)
    r = torch.rand(seq_length * n_head, head_size)
    td = torch.rand(seq_length * n_head, head_size)
    state = torch.rand(n_head, head_size, head_size)
    inputs = (k, v, r, state, td)
    print("!!! converting test graph")
    torch.onnx.export(gen_graph, inputs, "test_wkv.onnx", output_names=["output", "state"], opset_version=17)
    subprocess.call("qnn-onnx-converter -i test_wkv.onnx --input_layout state.1 NONTRIVIAL --op_package_config ../RwkvWkvOpPackageHTP.xml", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)

    os.path.exists("test_data_wkv") or os.mkdir("test_data_wkv")
    for i in range(len(inputs)):
        inputs[i].numpy().tofile(f"test_data_wkv/input{i}.bin")
    input_list_lines = [" ".join([f"test_data_wkv/input{i}.bin" for i in range(5)])]
    with open("input_list_wkv.txt", "w") as f:
        f.writelines(input_list_lines)
    print("!!! generating qnn model lib")
    subprocess.call("qnn-model-lib-generator -c test_wkv.cpp -b test_wkv.bin -t x86_64-linux-clang", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)
    print("!!! executing qnn-net-run")
    subprocess.call(f"qnn-net-run --input_list input_list_wkv.txt --model lib/x86_64-linux-clang/libtest_wkv.so --backend {qnn_sdk_root}/lib/x86_64-linux-clang/libQnnHtp.so --op_packages ../HTP/RwkvWkvOpPackage/build/x86_64-linux-clang/libQnnRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)
    qnn_output = torch.from_numpy(np.fromfile("output/Result_0/output.raw", dtype=np.float32)).reshape(seq_length * n_head, head_size)
    qnn_state = torch.from_numpy(np.fromfile("output/Result_0/state.raw", dtype=np.float32)).reshape(n_head, head_size, head_size)
    torch_output, torch_state = gen_graph(*inputs)
    if not torch.allclose(qnn_output, torch_output, rtol=1e-5, atol=1e-5):
        print(f"!!! wkv6 htp length={seq_length} output mismatch!!!")
    elif not torch.allclose(qnn_state, torch_state, rtol=1e-5, atol=1e-5):
        print(f"!!! wkv6 htp length={seq_length} state mismatch!!!")
    else:
        print(f"!!! wkv6 htp length={seq_length} output passed")

def test_wkv7_cpu(n_head, head_size, seq_length):
    print(f"!!! Testing wkv7 cpu length = {seq_length}")
    gen_graph = qnn_graph_v7(n_head, head_size)
    k = torch.rand(seq_length * n_head, head_size)
    v = torch.rand(seq_length * n_head, head_size)
    r = torch.rand(seq_length * n_head, head_size)
    w = torch.rand(seq_length * n_head, head_size)
    kk = torch.nn.functional.normalize(torch.rand(seq_length * n_head, head_size), p=2, dim=-1)
    a = torch.rand(seq_length * n_head, head_size)
    state = torch.rand(n_head, head_size, head_size)
    inputs = (r, w, k, v, -kk, kk * a, state)
    print("!!! converting test graph")
    torch.onnx.export(gen_graph, inputs, "test_wkv.onnx", output_names=["output", "state"], opset_version=17)
    subprocess.call("qnn-onnx-converter -i test_wkv.onnx --input_layout state.1 NONTRIVIAL --op_package_config ../RwkvWkvOpPackageCPU.xml", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)

    os.path.exists("test_data_wkv") or os.mkdir("test_data_wkv")
    for i in range(len(inputs)):
        inputs[i].numpy().tofile(f"test_data_wkv/input{i}.bin")
    input_list_lines = [" ".join([f"test_data_wkv/input{i}.bin" for i in range(len(inputs))])]
    with open("input_list_wkv.txt", "w") as f:
        f.writelines(input_list_lines)
    print("!!! generating qnn model lib")
    subprocess.call("qnn-model-lib-generator -c test_wkv.cpp -t x86_64-linux-clang", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)
    print("!!! executing qnn-net-run")
    subprocess.call(f"qnn-net-run --input_list input_list_wkv.txt --model lib/x86_64-linux-clang/libqnn_model.so --backend {qnn_sdk_root}/lib/x86_64-linux-clang/libQnnCpu.so --op_packages ../CPU/RwkvWkvOpPackage/libs/x86_64-linux-clang/libRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider", stdout=None, stderr=None, shell=True)
    qnn_output = torch.from_numpy(np.fromfile("output/Result_0/output.raw", dtype=np.float32)).reshape(seq_length * n_head, head_size)
    qnn_state = torch.from_numpy(np.fromfile("output/Result_0/state.raw", dtype=np.float32)).reshape(n_head, head_size, head_size)
    torch_output, torch_state = gen_graph(*inputs)
    if not torch.allclose(qnn_output, torch_output, rtol=1e-5, atol=1e-5):
        print(f"!!! wkv7 cpu length={seq_length} output mismatch!!!")
        print(qnn_output)
        print(torch_output)
    elif not torch.allclose(qnn_state, torch_state, rtol=1e-5, atol=1e-5):
        print(f"!!! wkv7 cpu length={seq_length} state mismatch!!!")
    else:
        print(f"!!! wkv7 cpu length={seq_length} output passed")

def test_wkv7_htp(n_head, head_size, seq_length):
    print(f"!!! Testing wkv7 htp length = {seq_length}")
    gen_graph = qnn_graph_v7(n_head, head_size)
    k = torch.rand(seq_length * n_head, head_size)
    v = torch.rand(seq_length * n_head, head_size)
    r = torch.rand(seq_length * n_head, head_size)
    w = torch.rand(seq_length * n_head, head_size)
    a = torch.rand(seq_length * n_head, head_size)
    b = torch.rand(seq_length * n_head, head_size)
    state = torch.rand(n_head, head_size, head_size)
    inputs = (r, w, k, v, a, b, state)
    print("!!! converting test graph")
    torch.onnx.export(gen_graph, inputs, "test_wkv.onnx", output_names=["output", "state"], opset_version=17)
    subprocess.call("qnn-onnx-converter -i test_wkv.onnx --input_layout state.1 NONTRIVIAL --op_package_config ../RwkvWkvOpPackageHTP.xml", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)

    os.path.exists("test_data_wkv") or os.mkdir("test_data_wkv")
    for i in range(len(inputs)):
        inputs[i].numpy().tofile(f"test_data_wkv/input{i}.bin")
    input_list_lines = [" ".join([f"test_data_wkv/input{i}.bin" for i in range(len(inputs))])]
    with open("input_list_wkv.txt", "w") as f:
        f.writelines(input_list_lines)
    print("!!! generating qnn model lib")
    subprocess.call("qnn-model-lib-generator -c test_wkv.cpp -t x86_64-linux-clang", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)
    print("!!! executing qnn-net-run")
    subprocess.call(f"qnn-net-run --input_list input_list_wkv.txt --model lib/x86_64-linux-clang/libqnn_model.so --backend {qnn_sdk_root}/lib/x86_64-linux-clang/libQnnHtp.so --op_packages ../HTP/RwkvWkvOpPackage/build/x86_64-linux-clang/libQnnRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)
    qnn_output = torch.from_numpy(np.fromfile("output/Result_0/output.raw", dtype=np.float32)).reshape(seq_length * n_head, head_size)
    qnn_state = torch.from_numpy(np.fromfile("output/Result_0/state.raw", dtype=np.float32)).reshape(n_head, head_size, head_size)
    torch_output, torch_state = gen_graph(*inputs)
    if not torch.allclose(qnn_output, torch_output, rtol=1e-5, atol=1e-5):
        print(f"!!! wkv7 htp length={seq_length} output mismatch!!!")
    elif not torch.allclose(qnn_state, torch_state, rtol=1e-5, atol=1e-5):
        print(f"!!! wkv7 htp length={seq_length} state mismatch!!!")
    else:
        print(f"!!! wkv7 htp length={seq_length} output passed")

print("=========== testing wkv6 ===========")
test_wkv6_cpu(n_head, head_size, 1)
test_wkv6_cpu(n_head, head_size, 32)
test_wkv6_cpu(n_head, head_size, 64)
test_wkv6_cpu(n_head, head_size, 128)

test_wkv6_htp(n_head, head_size, 1)
test_wkv6_htp(n_head, head_size, 32)
test_wkv6_htp(n_head, head_size, 64)
test_wkv6_htp(n_head, head_size, 128)

print("=========== testing wkv7 ===========")
test_wkv7_cpu(n_head, head_size, 1)
test_wkv7_cpu(n_head, head_size, 32)
test_wkv7_cpu(n_head, head_size, 64)
test_wkv7_cpu(n_head, head_size, 128)

test_wkv7_htp(n_head, head_size, 1)
test_wkv7_htp(n_head, head_size, 32)
test_wkv7_htp(n_head, head_size, 64)
test_wkv7_htp(n_head, head_size, 128)
