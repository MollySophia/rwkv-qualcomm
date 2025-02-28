//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "ClientBuffer.hpp"
#include "QnnTypeMacros.hpp"

void* ClientBuffer::getBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_WARN("getBuffer: received a null pointer to a tensor");
    return nullptr;
  }
  return QNN_TENSOR_GET_CLIENT_BUF(tensor).data;
}

size_t ClientBuffer::getBufferSize(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_WARN("getBufferSize: received a null pointer to a tensor");
    return 0;
  }
  return QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize;
};

bool ClientBuffer::allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_RAW);
  Qnn_ClientBuffer_t clientBuffer;
  clientBuffer.data = malloc(tensorDataSize);
  if (nullptr == clientBuffer.data) {
    QNN_ERROR("mem alloc failed for clientBuffer.data");
    return false;
  }
  clientBuffer.dataSize = tensorDataSize;
  QNN_TENSOR_SET_CLIENT_BUF(tensor, clientBuffer);
  return true;
}

bool ClientBuffer::freeTensorBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  if (QNN_TENSOR_GET_CLIENT_BUF(tensor).data) {
    if (m_sameMemoryFreeTensors.find(tensor) == m_sameMemoryFreeTensors.end()) {
      free(QNN_TENSOR_GET_CLIENT_BUF(tensor).data);
    }
    QNN_TENSOR_SET_CLIENT_BUF(tensor, Qnn_ClientBuffer_t({nullptr, 0u}));
    QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_UNDEFINED);
  }
  return true;
}

bool ClientBuffer::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    QNN_ERROR("Received nullptr");
    return false;
  }
  if (false == freeTensorBuffer(dest)) {
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_CLIENT_BUF(dest, QNN_TENSOR_GET_CLIENT_BUF(src));
  m_sameMemoryFreeTensors.insert(dest);
  return true;
}

bool ClientBuffer::useExternalMemory(Qnn_Tensor_t* dest, void* extMem) {
  if (nullptr == dest || nullptr == extMem) {
    QNN_ERROR("Received nullptr");
    return false;
  }

  Qnn_ClientBuffer_t clientBuffer;
  clientBuffer.data     = extMem;
  clientBuffer.dataSize = QNN_TENSOR_GET_CLIENT_BUF(dest).dataSize;
  if (false == freeTensorBuffer(dest)) {
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSORMEMTYPE_RAW);
  QNN_TENSOR_SET_CLIENT_BUF(dest, clientBuffer);
  m_sameMemoryFreeTensors.insert(dest);
  return true;
}

void* ClientBuffer::allocateTensorFusedBuffer(uint64_t bufferSize, int32_t* fd) { return nullptr; }

bool ClientBuffer::mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                                        size_t tensorDataSize,
                                        int32_t fd,
                                        uint32_t offset,
                                        uint64_t totalBufferSize,
                                        void* memPointer,
                                        Qnn_ContextHandle_t contextHandle) {
  return false;
}

bool ClientBuffer::deregisterTensorFusedBuffer(Qnn_Tensor_t* tensor) { return false; }

void ClientBuffer::freeFusedBuffers() {}

size_t ClientBuffer::getOffset(Qnn_Tensor_t* tensor) { return 0; }

size_t ClientBuffer::getTotalBufferSize(Qnn_Tensor_t* tensor) { return 0; }