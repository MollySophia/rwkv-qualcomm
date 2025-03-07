```
rm -rf trace_output
./qnn-net-run --profiling_level detailed --profiling_option optrace --output_data_type float_and_native --retrieve_context RWKV-x070-World-1.5B-v3-20250127-ctx4096.bin --backend libQnnHtp.so --input_list ./input_list.txt --output_dir ./trace_output --log_level info --perf_profile burst --io_tensor_mem_handle_type=ion
# or with customop:
./qnn-net-run --profiling_level detailed --profiling_option optrace --output_data_type float_and_native --retrieve_context RWKV-x070-World-1.5B-v3-20250127-ctx4096.bin --backend libQnnHtp.so --input_list ./input_list.txt --output_dir ./trace_output --log_level info --perf_profile burst --io_tensor_mem_handle_type=ion --op_packages libQnnRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider
```

```
adb pull /data/local/tmp/rwkv/trace_output
qnn-profile-viewer --reader $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so --input_log ./trace_output/qnn-profiling-data_0.log --schematic ./RWKV-x070-World-1.5B-v3-20250127-ctx4096_schematic.bin --output ./chrometrace.json
```