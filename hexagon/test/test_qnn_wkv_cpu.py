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
wkv_chunk_size = 32

cmd_stdout = subprocess.DEVNULL
cmd_stderr = subprocess.DEVNULL
# cmd_stdout = None
# cmd_stderr = None

class qnn_graph(torch.nn.Module):
    def __init__(self, chunk_size, n_head, head_size):
        super().__init__()
        if chunk_size == 1:
            self.wkv_func = torch.ops.rwkv.wkv
        elif chunk_size == wkv_chunk_size:
            self.wkv_func = torch.ops.rwkv.wkv_chunk
        else:
            assert False
        
        self.tf = torch.rand(n_head, head_size, 1)
    
    def forward(self, k, v, r, state, td):
        return self.wkv_func(k, v, r, state, self.tf, td)

def onnx_custom_wkv(g, k, v, r, state2, time_first, time_decay):
    out1, out2 = g.op("rwkv::wkv", k, v, r, state2, time_first, time_decay, outputs=2)
    return out1.setType(k.type().with_dtype(torch.float32).with_sizes([1, n_head, 1, head_size])),\
        out2.setType(k.type().with_dtype(torch.float32).with_sizes([n_head, head_size, head_size]))
def onnx_custom_wkv_chunk(g, k, v, r, state2, time_first, time_decay):
    out1, out2 = g.op("rwkv::wkv_chunk", k, v, r, state2, time_first, time_decay, outputs=2)
    return out1.setType(k.type().with_dtype(torch.float32).with_sizes([wkv_chunk_size, n_head, 1, head_size])),\
        out2.setType(k.type().with_dtype(torch.float32).with_sizes([n_head, head_size, head_size]))
from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic("rwkv::wkv", onnx_custom_wkv, 9)
register_custom_op_symbolic("rwkv::wkv_chunk", onnx_custom_wkv_chunk, 9)

print("Testing wkv length = 1")
gen_graph = qnn_graph(1, n_head, head_size)
k = torch.rand(n_head, 1, head_size)
v = torch.rand(n_head, head_size, 1)
r = torch.rand(n_head, head_size, 1)
td = torch.rand(n_head, head_size, 1)
state = torch.rand(n_head, head_size, head_size)
inputs = (k, v, r, state, td)
print("converting test graph")
torch.onnx.export(gen_graph, inputs, "test_wkv.onnx", output_names=["output", "state"], opset_version=17)
subprocess.call("qnn-onnx-converter -i test_wkv.onnx --op_package_config ../RwkvWkvOpPackageCPU.xml", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)

os.path.exists("test_data_wkv") or os.mkdir("test_data_wkv")
for i in range(len(inputs)):
    inputs[i].numpy().tofile(f"test_data_wkv/input{i}.bin")
input_list_lines = [" ".join([f"test_data_wkv/input{i}.bin" for i in range(5)])]
with open("input_list_wkv.txt", "w") as f:
    f.writelines(input_list_lines)
print("generating qnn model lib")
subprocess.call("qnn-model-lib-generator -c test_wkv.cpp -b test_wkv.bin -t x86_64-linux-clang", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)
print("executing qnn-net-run")
subprocess.call("qnn-net-run --input_list input_list_wkv.txt --model lib/x86_64-linux-clang/libtest_wkv.so --backend /opt/qcom/aistack/qairt/2.26.0.240828/lib/x86_64-linux-clang/libQnnCpu.so --op_packages ../CPU/RwkvWkvOpPackage/libs/x86_64-linux-clang/libRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider", stdout=None, stderr=None, shell=True)
qnn_output = torch.from_numpy(np.fromfile("output/Result_0/output.raw", dtype=np.float32)).reshape(1, n_head, 1, head_size)
qnn_state = torch.from_numpy(np.fromfile("output/Result_0/state.raw", dtype=np.float32)).reshape(n_head, head_size, head_size)
torch_output, torch_state = gen_graph(*inputs)
if not torch.allclose(qnn_output, torch_output, rtol=1e-5, atol=1e-5) or not torch.allclose(qnn_state, torch_state, rtol=1e-5, atol=1e-5):
    print("!!!wkv length=1 output mismatch!!!")
else:
    print("wkv length=1 output passed")

print("\n\nTesting wkv length = " + str(wkv_chunk_size))
chunk_graph = qnn_graph(wkv_chunk_size, n_head, head_size)
k = torch.rand(wkv_chunk_size, n_head, 1, head_size)
v = torch.rand(wkv_chunk_size, n_head, head_size, 1)
r = torch.rand(wkv_chunk_size, n_head, head_size, 1)
td = torch.rand(wkv_chunk_size, n_head, head_size, 1)
state = torch.rand(n_head, head_size, head_size)
inputs = (k, v, r, state, td)
print("converting test graph")
torch.onnx.export(chunk_graph, inputs, "test_wkv_chunk.onnx", output_names=["output", "state"], opset_version=9)
subprocess.call("qnn-onnx-converter -i test_wkv_chunk.onnx --op_package_config ../RwkvWkvOpPackageCPU.xml", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)
os.path.exists("test_data_wkv_chunk") or os.mkdir("test_data_wkv_chunk")
for i in range(len(inputs)):
    inputs[i].numpy().tofile(f"test_data_wkv_chunk/input{i}.bin")
input_list_lines = [" ".join([f"test_data_wkv_chunk/input{i}.bin" for i in range(5)])]
with open("input_list_wkv_chunk.txt", "w") as f:
    f.writelines(input_list_lines)
print("generating qnn model lib")
subprocess.call("qnn-model-lib-generator -c test_wkv_chunk.cpp -b test_wkv_chunk.bin -t x86_64-linux-clang", stdout=cmd_stdout, stderr=cmd_stderr, shell=True)
print("executing qnn-net-run")
subprocess.call("qnn-net-run --input_list input_list_wkv_chunk.txt --model lib/x86_64-linux-clang/libtest_wkv_chunk.so --backend /opt/qcom/aistack/qairt/2.26.0.240828/lib/x86_64-linux-clang/libQnnCpu.so --op_packages ../CPU/RwkvWkvOpPackage/libs/x86_64-linux-clang/libRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider", stdout=None, stderr=None, shell=True)
qnn_output = torch.from_numpy(np.fromfile("output/Result_0/output.raw", dtype=np.float32)).reshape(wkv_chunk_size, n_head, 1, head_size)
qnn_state = torch.from_numpy(np.fromfile("output/Result_0/state.raw", dtype=np.float32)).reshape(n_head, head_size, head_size)
torch_output, torch_state = chunk_graph(*inputs)
if not torch.allclose(qnn_output, torch_output, rtol=1e-5, atol=1e-5) or not torch.allclose(qnn_state, torch_state, rtol=1e-5, atol=1e-5):
    print(f"!!!wkv length={wkv_chunk_size} output mismatch!!!")
else:
    print(f"wkv length={wkv_chunk_size} output passed")
