```
rm -rf trace_output
./qnn-net-run --profiling_level detailed --profiling_option optrace --output_data_type float_and_native --retrieve_context RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk2of2.bin --backend libQnnHtp.so --input_list ./input_list_chunk1_test.txt --output_dir ./trace_output --log_level info --perf_profile burst
```

```
adb pull /data/local/tmp/rwkv/trace_output
qnn-profile-viewer --reader $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so --input_log ./trace_output/qnn-profiling-data_0.log --schematic ./RWKV_x060_World_1B6_v2_1_20240328_ctx4096_chunk2of2_schematic.bin --output ./chrometrace.json
```