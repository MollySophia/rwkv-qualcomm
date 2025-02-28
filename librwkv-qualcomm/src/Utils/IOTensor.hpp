//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IBufferAlloc.hpp"
#include "Logger.hpp"
#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnProperty.h"
#include "QnnTensor.h"
#include "QnnTypeDef.hpp"
#include "QnnTypes.h"
enum class BufferAlloc {
  DEFAULT,        // malloc based allocator
  SHARED_BUFFER,  // shared buffer allocator; actual allocator depends on the platform
  DMABUF,         // dma buffer allocator
  INVALID
};
class IBufferAlloc;
class IOTensor {
 public:
  IOTensor(BufferAlloc bufferAllocIn            = BufferAlloc::DEFAULT,
           QNN_INTERFACE_VER_TYPE* qnnInterface = nullptr);

  ~IOTensor();

  bool initialize(Qnn_ContextHandle_t contextHandle = nullptr);

  bool setupInputTensors(Qnn_Tensor_t** inputs,
                         std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                         const GraphInfo_t& graphInfo,
                         std::unordered_map<std::string, size_t>& inputTensorsSize,
                         Qnn_ContextHandle_t contextHandle,
                         bool skipBufferAllocation = false);

  bool setupOutputTensors(Qnn_Tensor_t** outputs,
                          std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                          const GraphInfo_t& graphInfo,
                          std::unordered_map<std::string, size_t>& outputTensorsSize,
                          Qnn_ContextHandle_t contextHandle,
                          bool skipBufferAllocation = false);

  bool setupInputWithSharedTensors(
      Qnn_Tensor_t** tensors,
      std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
      const GraphInfo_t& graphInfo,
      std::unordered_map<std::string, size_t>& tensorsSize,
      Qnn_ContextHandle_t contextHandle,
      std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap);

  bool setupOutputWithSharedTensors(
      Qnn_Tensor_t** tensors,
      std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
      const GraphInfo_t& graphInfo,
      std::unordered_map<std::string, size_t>& tensorsSize,
      Qnn_ContextHandle_t contextHandle,
      std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap);

  bool tearDownTensors(Qnn_Tensor_t* tensors, uint32_t tensorCount);

  bool tearDownTensors(std::vector<Qnn_Tensor_t*>& tensors, uint32_t tensorCount);
  bool tearDownTensors(std::vector<Qnn_Tensor_t>& tensors);
  bool tearDownTensors(std::unordered_map<std::string, Qnn_Tensor_t*>& tensors,
                       std::unordered_map<std::string, uint32_t>& tensorCountMap);
  bool tearDownTensors(std::vector<std::unordered_map<std::string, Qnn_Tensor_t*>>& tensors,
                       std::unordered_map<std::string, uint32_t>& tensorCountMap);

  bool tearDownTensors(const GraphInfo_t* graph_info) {
    bool status = true;
    if (!tearDownTensors(graph_info->inputTensors, graph_info->numInputTensors)) {
      status = false;
      QNN_ERROR("Failed to tear down input tensors for graph %s", graph_info->graphName);
    }

    if (!tearDownTensors(graph_info->outputTensors, graph_info->numOutputTensors)) {
      status = false;
      QNN_ERROR("Failed to tear down output tensors for graph %s", graph_info->graphName);
    }
    return status;
  }

  void* getBuffer(Qnn_Tensor_t* tensor) { return m_bufferManager->getBuffer(tensor); };

  int getFd(Qnn_Tensor_t* tensor) { return m_bufferManager->getFd(tensor); };

  size_t getOffset(Qnn_Tensor_t* tensor) { return m_bufferManager->getOffset(tensor); };

  size_t getBufferSize(Qnn_Tensor_t* tensor) { return m_bufferManager->getBufferSize(tensor); };

  size_t getTotalBufferSize(Qnn_Tensor_t* tensor) {
    return m_bufferManager->getTotalBufferSize(tensor);
  }

  void* allocateTensorFusedBuffer(uint64_t bufferSize, int32_t* fd) {
    return m_bufferManager->allocateTensorFusedBuffer(bufferSize, fd);
  }

  bool allocateBuffers(const std::map<int, std::map<std::string, size_t>>& allocs_per_chunk,
                       std::map<std::string, std::pair<int, size_t>>& tensor_offsets) {
    return m_bufferManager->allocateBuffers(allocs_per_chunk, tensor_offsets);
  }

  bool mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                            size_t tensorDataSize,
                            int32_t fd,
                            uint32_t offset,
                            uint64_t totalBufferSize,
                            void* memPointer,
                            Qnn_ContextHandle_t contextHandle) {
    return m_bufferManager->mapFusedBufferOffset(
        tensor, tensorDataSize, fd, offset, totalBufferSize, memPointer, contextHandle);
  }

  bool mapFusedBufferOffset(
      GraphInfo_t* graph_info,
      Qnn_ContextHandle_t context_handle,
      const std::map<std::string, std::tuple<int, size_t, size_t>>& graph_allocs);

  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
    return m_bufferManager->useSameMemory(dest, src);
  }

  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src, int offset) {
    return m_bufferManager->useSameMemory(dest, src, offset);
  }

  bool useExternalMemory(Qnn_Tensor_t* dest, void* extMem) {
    return m_bufferManager->useExternalMemory(dest, extMem);
  }

  BufferAlloc getBufferAllocType() { return m_bufferAlloc; }

  std::unordered_set<void*>& getFreeTensorsPointerSet() { return m_freeTensorsPointerSet; }

  // Functions to sync memory buffers for Read/Write using DmaBuf.
  bool beforeWriteToBuffer(Qnn_Tensor_t* tensor) {
    return m_bufferManager->beforeWriteToBuffer(tensor);
  }
  bool afterWriteToBuffer(Qnn_Tensor_t* tensor) {
    return m_bufferManager->afterWriteToBuffer(tensor);
  }
  bool beforeReadFromBuffer(Qnn_Tensor_t* tensor) {
    return m_bufferManager->beforeReadFromBuffer(tensor);
  }
  bool afterReadFromBuffer(Qnn_Tensor_t* tensor) {
    return m_bufferManager->afterReadFromBuffer(tensor);
  }

 private:
  BufferAlloc m_bufferAlloc;
  QNN_INTERFACE_VER_TYPE* m_qnnInterface;
  std::unique_ptr<IBufferAlloc> m_bufferManager;
  std::unordered_set<void*> m_freeTensorsPointerSet;

  // There seems to be a race condition in mapFusedBufferOffset because we are
  // calling it from multiple threads. Maybe memRegister/memDeRegister is not thread-safe
  // Until I figure this out, adding a temporary lock here. TODO: Fix and remove this!
  std::mutex _tmp_lock;

  bool deepCopyQnnTensorInfo(Qnn_Tensor_t* dest, Qnn_Tensor_t* src);
  bool setupTensors(Qnn_Tensor_t** tensors,
                    std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                    uint32_t tensorCount,
                    TensorWrapper* tensorsInfo,
                    std::unordered_map<std::string, size_t>& tensorsSize,
                    Qnn_ContextHandle_t contextHandle,
                    bool skipBufferAllocation = false);
};