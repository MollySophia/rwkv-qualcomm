//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <stdlib.h>

#include <unordered_set>

#include "IBufferAlloc.hpp"
#include "Logger.hpp"

class ClientBuffer final : public IBufferAlloc {
 public:
  ClientBuffer(){};

  // Disable copy constructors, r-value referencing, etc
  ClientBuffer(const ClientBuffer&) = delete;

  ClientBuffer& operator=(const ClientBuffer&) = delete;

  ClientBuffer(ClientBuffer&&) = delete;

  ClientBuffer& operator=(ClientBuffer&&) = delete;

  bool initialize() override { return true; };

  void* getBuffer(Qnn_Tensor_t* tensor) override;

  int getFd(Qnn_Tensor_t* tensor) override {
    QNN_WARN("getFd: This is not ION memory");
    return -1;
  };

  size_t getOffset(Qnn_Tensor_t* tensor) override;
  size_t getBufferSize(Qnn_Tensor_t* tensor) override;
  size_t getTotalBufferSize(Qnn_Tensor_t* tensor) override;

  bool allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) override;

  bool freeTensorBuffer(Qnn_Tensor_t* tensor) override;

  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) override;
  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src, int offset) override { return false; }

  bool useExternalMemory(Qnn_Tensor_t* dest, void* extMem) override;

  void* allocateTensorFusedBuffer(uint64_t bufferSize, int32_t* fd) override;
  bool allocateBuffers(const std::map<int, std::map<std::string, size_t>>& allocs_per_chunk,
                       std::map<std::string, std::pair<int, size_t>>& tensor_offsets) override {
    return false;
  };

  bool mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                            size_t tensorDataSize,
                            int32_t fd,
                            uint32_t offset,
                            uint64_t totalBufferSize,
                            void* memPointer,
                            Qnn_ContextHandle_t contextHandle) override;
  bool deregisterTensorFusedBuffer(Qnn_Tensor_t* tensor) override;
  void freeFusedBuffers() override;

  bool mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                            int alloc_idx,
                            size_t offset,
                            Qnn_ContextHandle_t ctx,
                            size_t size) override {
    return false;
  }

  virtual ~ClientBuffer(){};

 private:
  std::unordered_set<Qnn_Tensor_t*> m_sameMemoryFreeTensors;
};
