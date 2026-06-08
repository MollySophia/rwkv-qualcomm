# RWKV QNN Cross-PD Shared Buffer Issue Notes

Date: 2026-05-18

## Background

This minimal reproduction uses the RWKV 7B QNN context binaries split into six embedding graphs:

- `chunk1of6`
- `chunk2of6`
- `chunk3of6`
- `chunk4of6`
- `chunk5of6`
- `chunk6of6`

The repro binary and QNN 2.42 Android libraries run on device `adb_mi17`, with files under:

```text
/data/local/tmp/rwkv
```

Local relevant paths:

```text
/home/molly/workspace/rwkv-qualcomm/onnx_files/output
/home/molly/workspace/rwkv-qualcomm/qnn_rwkv_7b_test
/home/molly/workspace/rwkv-qualcomm/fix1_qnn_backend.txt
```

## Symptom

Before the Qualcomm fix, the repro consistently executes graph 1 through graph 5, then fails when executing graph 6:

```text
Executing graph ... chunk6of6
QnnDsp <E> Graph ... chunk6of6 failed in execution with err 1003
eval_with_dummy_input failed, ret=4
SSR detected on device config {0,0,2}
```

Observed test result:

```text
/data/local/tmp/rwkv/qnn_rwkv_7b_min_repro_before
exit code: 3
```

## Root Cause

The model reuses buffers across graphs using `useSameMemory`.

Important reused tensors:

```text
out_chunk1          -> in_chunk2/3/4/5/6
v_first_out_chunk1  -> v_first_in_chunk2/3/4/5/6
```

The original `useSameMemory()` implementation only copies the source tensor's memory type and memory handle:

```cpp
QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
QNN_TENSOR_SET_MEM_HANDLE(dest, QNN_TENSOR_GET_MEM_HANDLE(src));
```

That means the reused input tensor may carry a memHandle that was registered in a different QNN context / process domain.

In this repro, QNN runtime logs show:

```text
PD 0 chosen for context 1
PD 0 chosen for context 2
PD 0 chosen for context 3
PD 0 chosen for context 4
PD 0 chosen for context 5
PD 2 chosen for context 6
```

So graph 6 runs in PD2, but `in_chunk6` and `v_first_in_chunk6` reuse buffers whose handles were originally registered in PD0. PD2 does not have the proper SMMU/TLB mapping for those buffers, leading to the graph execution failure and SSR.

Note: the app does not query PD assignment directly. The PD split above is observed from QNN debug logs and Qualcomm's analysis.

## Qualcomm Fix1

Qualcomm's attached fix is saved locally as:

```text
/home/molly/workspace/rwkv-qualcomm/fix1_qnn_backend.txt
```

Core idea:

1. Let `setupInputWithSharedTensors()` reuse the original buffer first.
2. For the last embedding graph only, find input tensors that share the same fd as:
   - `hiddenStateTensor`
   - `vFirstTensor`
3. Re-register those shared buffers against the last graph's context.
4. Replace the input tensor's memHandle with the new handle.

Key effect:

```text
same fd/shared buffer, but a memHandle registered in graph 6's context/PD
```

Observed test result:

```text
/data/local/tmp/rwkv/qnn_rwkv_7b_min_repro_fix1
exit code: 0
```

Important log lines:

```text
Re-registered cross-PD input tensor in_chunk6 with last context
Re-registered cross-PD input tensor v_first_in_chunk6 with last context
```

## Fix1 Limitation

Fix1 relies on a structural assumption:

```cpp
graph_id == qnnEmbdGraphsCount - 1
```

It does not dynamically detect whether a graph is actually placed in a different PD. It works for this repro because graph 6 is the one placed in PD2.

If future QNN placement changes and another graph crosses PD, fix1 may miss it.

## Fix2 Experiment

I created a generalized fix2 variant here:

```text
/home/molly/workspace/rwkv-qualcomm/fix2_qnn_backend.cpp
/home/molly/workspace/rwkv-qualcomm/fix2_qnn_backend.h
```

The first attempt re-registered all shared input tensors for every non-first graph. That failed during graph 6 tensor initialization:

```text
Duplicate Memory Handle found
Memory handle for custom shared buffer exists already
Failed to register memHandles for context 6 on pdId 2
```

Reason: state tensors already have their own valid registration pattern and should not be blindly re-registered.

The successful fix2 version narrows the generalized re-registration to cross-chunk activation inputs only:

```text
in_chunk2/3/4/5/6
v_first_in_chunk2/3/4/5/6
```

It applies to every `graph_id > 0`, not just the last graph.

It also stores the newly registered memHandles and explicitly deregisters them in `release_model()` before freeing contexts.

Observed test result:

```text
/data/local/tmp/rwkv/qnn_rwkv_7b_min_repro_fix2
exit code: 0
```

Important log lines:

```text
Re-registered shared input tensor in_chunk2 for graph 1 with its context
Re-registered shared input tensor v_first_in_chunk2 for graph 1 with its context
...
Re-registered shared input tensor in_chunk6 for graph 5 with its context
Re-registered shared input tensor v_first_in_chunk6 for graph 5 with its context
Executing graph ... chunk6of6
```

No `err 1003`, `DspTransport`, or `SSR` error appeared in the final fix2 run.

## Current Recommendation

For the minimal repro, both fix1 and the narrowed fix2 pass.

For a more robust app-side fix, prefer the narrowed fix2 approach:

- Do not assume only the last graph can be placed in another PD.
- Re-register only known cross-chunk activation buffers (`in_chunk*`, `v_first_in_chunk*`) against each graph's own context.
- Do not blindly re-register every shared input tensor, because state tensors can trigger duplicate-handle errors.
- Track and deregister newly created memHandles during model release.

## Test Artifacts

Local logs:

```text
/tmp/rwkv_minrepro_before.log
/tmp/rwkv_minrepro_fix1.log
/tmp/rwkv_minrepro_fix2.log
```

Remote binaries:

```text
/data/local/tmp/rwkv/qnn_rwkv_7b_min_repro_before
/data/local/tmp/rwkv/qnn_rwkv_7b_min_repro_fix1
/data/local/tmp/rwkv/qnn_rwkv_7b_min_repro_fix2
```
