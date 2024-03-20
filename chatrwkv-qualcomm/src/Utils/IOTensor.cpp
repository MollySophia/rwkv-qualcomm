//==============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "half.hpp"
#include "DataUtil.hpp"
#include "IOTensor.hpp"
#include "Logger.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#include "PAL/StringOp.hpp"
#include "QnnTypeMacros.hpp"

using namespace qnn;
using namespace qnn::tools;

// Helper method to read data from files to a buffer.
iotensor::PopulateInputTensorsRetType_t iotensor::IOTensor::readDataAndAllocateBuffer(
    const std::vector<std::string>& filePaths,
    const size_t filePathsIndexOffset,
    const bool loopBackToStart,
    std::vector<size_t> dims,
    Qnn_DataType_t dataType,
    uint8_t** bufferToCopy) {
  StatusCode returnStatus  = StatusCode::SUCCESS;
  *bufferToCopy            = nullptr;
  returnStatus             = allocateBuffer(bufferToCopy, dims, dataType);
  size_t numFilesPopulated = 0;
  size_t batchSize         = 0;
  datautil::StatusCode status;
  std::tie(status, numFilesPopulated, batchSize) =
      datautil::readBatchData(filePaths,
                              filePathsIndexOffset,
                              loopBackToStart,
                              dims,
                              dataType,
                              reinterpret_cast<uint8_t*>(*bufferToCopy));
  if (datautil::StatusCode::SUCCESS != status) {
    QNN_ERROR("Failure in datautil::readBatchData");
    returnStatus = StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS != returnStatus) {
    if (nullptr != *bufferToCopy) {
      free(*bufferToCopy);
      *bufferToCopy = nullptr;
    }
  }
  return {returnStatus, numFilesPopulated, batchSize};
}

// Helper method to copy a float buffer, quantize it, and copy
// it to a tensor (Qnn_Tensor_t) buffer.
iotensor::StatusCode iotensor::IOTensor::copyFromFloatToNative(float* floatBuffer,
                                                               Qnn_Tensor_t* tensor) {
  if (nullptr == floatBuffer || nullptr == tensor) {
    QNN_ERROR("copyFromFloatToNative(): received a nullptr");
    return StatusCode::FAILURE;
  }

  StatusCode returnStatus = StatusCode::SUCCESS;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor));

  switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      datautil::floatToTfN<uint8_t>(static_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    floatBuffer,
                                    QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
                                    QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
                                    datautil::calculateElementCount(dims));
      break;

    case QNN_DATATYPE_UFIXED_POINT_16:
      datautil::floatToTfN<uint16_t>(static_cast<uint16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                     floatBuffer,
                                     QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
                                     QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
                                     datautil::calculateElementCount(dims));
      break;

    case QNN_DATATYPE_UINT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint8_t>(
              static_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              floatBuffer,
              datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint16_t>(
              static_cast<uint16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              floatBuffer,
              datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint32_t>(
              static_cast<uint32_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              floatBuffer,
              datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int8_t>(
              static_cast<int8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              floatBuffer,
              datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int16_t>(
              static_cast<int16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              floatBuffer,
              datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int32_t>(
              static_cast<int32_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              floatBuffer,
              datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_BOOL_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint8_t>(
              static_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              floatBuffer,
              datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<bool>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    default:
      QNN_ERROR("Datatype not supported yet!");
      returnStatus = StatusCode::FAILURE;
      break;
  }
  return returnStatus;
}

// Helper method to populate an input tensor in the graph during execution.
// It relies on reading data from files provided during app creation.
iotensor::PopulateInputTensorsRetType_t iotensor::IOTensor::populateInputTensor(
    const std::vector<std::string>& filePaths,
    const size_t filePathsIndexOffset,
    const bool loopBackToStart,
    Qnn_Tensor_t* input,
    iotensor::InputDataType inputDataType) {
  if (nullptr == input) {
    QNN_ERROR("input is nullptr");
    return {StatusCode::FAILURE, 0, 0};
  }

  auto returnStatus        = StatusCode::SUCCESS;
  size_t numFilesPopulated = 0;
  size_t batchSize         = 0;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(input), QNN_TENSOR_GET_RANK(input));

  if (inputDataType == InputDataType::FLOAT &&
      QNN_TENSOR_GET_DATA_TYPE(input) != QNN_DATATYPE_FLOAT_32) {
    uint8_t* fileToBuffer = nullptr;
    std::tie(returnStatus, numFilesPopulated, batchSize) =
        readDataAndAllocateBuffer(filePaths,
                                  filePathsIndexOffset,
                                  loopBackToStart,
                                  dims,
                                  QNN_DATATYPE_FLOAT_32,
                                  &fileToBuffer);
    if (StatusCode::SUCCESS == returnStatus) {
      QNN_DEBUG("readDataFromFileToBuffer successful");
      returnStatus = copyFromFloatToNative(reinterpret_cast<float*>(fileToBuffer), input);
    }
    if (nullptr != fileToBuffer) {
      free(fileToBuffer);
      fileToBuffer = nullptr;
    }
  } else {
    datautil::StatusCode status;
    std::tie(status, numFilesPopulated, batchSize) =
        datautil::readBatchData(filePaths,
                                filePathsIndexOffset,
                                loopBackToStart,
                                dims,
                                QNN_TENSOR_GET_DATA_TYPE(input),
                                static_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(input).data));
    if (datautil::StatusCode::SUCCESS != status) {
      QNN_ERROR("Failure in datautil::readBatchData");
      returnStatus = StatusCode::FAILURE;
    }
  }
  return {returnStatus, numFilesPopulated, batchSize};
}

// Helper method to populate all input tensors during execution.
iotensor::PopulateInputTensorsRetType_t iotensor::IOTensor::populateInputTensors(
    uint32_t graphIdx,
    const std::vector<std::vector<std::string>>& filePathsVector,
    const size_t filePathsIndexOffset,
    const bool loopBackToStart,
    const std::unordered_map<std::string, uint32_t>& inputNameToIndex,
    Qnn_Tensor_t* inputs,
    qnn_wrapper_api::GraphInfo_t graphInfo,
    iotensor::InputDataType inputDataType) {
  QNN_DEBUG("populateInputTensors() graphIndx %d", graphIdx);
  if (nullptr == inputs) {
    QNN_ERROR("inputs is nullptr");
    return {StatusCode::FAILURE, 0, 0};
  }
  auto inputCount = graphInfo.numInputTensors;
  if (filePathsVector.size() != inputCount) {
    QNN_ERROR(
        "Incorrect amount of Input files for graphIdx: %d. Expected: %d, "
        "received: %d",
        graphIdx,
        inputCount,
        filePathsVector.size());
    return {StatusCode::FAILURE, 0, 0};
  }
  size_t numFilesPopulated = 0;
  size_t numBatchSize      = 0;
  for (size_t inputIdx = 0; inputIdx < inputCount; inputIdx++) {
    size_t inputNameIdx = inputIdx;
    QNN_DEBUG("index = %d input column index = %d", inputIdx, inputNameIdx);
    std::string inputNodeName;
    if (QNN_TENSOR_GET_NAME(graphInfo.inputTensors[inputIdx]))
      inputNodeName = QNN_TENSOR_GET_NAME(graphInfo.inputTensors[inputIdx]);
    if (!inputNodeName.empty() && inputNameToIndex.find(inputNodeName) != inputNameToIndex.end()) {
      inputNameIdx = inputNameToIndex.at(inputNodeName);
    }
    StatusCode returnStatus;
    size_t currentInputNumFilesPopulated = 0;
    size_t currentInputNumBatchSize      = 0;
    std::tie(returnStatus, currentInputNumFilesPopulated, currentInputNumBatchSize) =
        populateInputTensor(filePathsVector[inputNameIdx],
                            filePathsIndexOffset,
                            loopBackToStart,
                            &(inputs[inputIdx]),
                            inputDataType);
    if (StatusCode::SUCCESS != returnStatus) {
      QNN_ERROR("populateInputTensorFromFiles failed for input %s with index %d",
                inputNodeName.c_str(),
                inputIdx);
      return {StatusCode::FAILURE, currentInputNumFilesPopulated, currentInputNumBatchSize};
    }
    if (inputIdx == 0) {
      numFilesPopulated = currentInputNumFilesPopulated;
      numBatchSize      = currentInputNumBatchSize;
    } else {
      if (numFilesPopulated != currentInputNumFilesPopulated ||
          numBatchSize != currentInputNumBatchSize) {
        QNN_ERROR(
            "Current input tensor with name: %s with index %d files populated = %d, batch size = %d"
            " does not match with expected files populated = %d, batch size = %d",
            inputNodeName.c_str(),
            inputIdx,
            currentInputNumFilesPopulated,
            currentInputNumBatchSize,
            numFilesPopulated,
            numBatchSize);
        return {StatusCode::FAILURE, numFilesPopulated, numBatchSize};
      }
    }
  }
  return {StatusCode::SUCCESS, numFilesPopulated, numBatchSize};
}

// Setup details for Qnn_Tensor_t for execution
// based on information in Qnn_TensorWrapper_t provided by model.so.
iotensor::StatusCode iotensor::IOTensor::setupTensors(Qnn_Tensor_t** tensors,
                                                      uint32_t tensorCount,
                                                      Qnn_Tensor_t* tensorWrappers) {
  if (nullptr == tensorWrappers) {
    QNN_ERROR("tensorWrappers is nullptr");
    return StatusCode::FAILURE;
  }
  if (0 == tensorCount) {
    QNN_INFO("tensor count is 0. Nothing to setup.");
    return StatusCode::SUCCESS;
  }
  auto returnStatus = StatusCode::SUCCESS;
  *tensors          = (Qnn_Tensor_t*)calloc(1, tensorCount * sizeof(Qnn_Tensor_t));
  if (nullptr == *tensors) {
    QNN_ERROR("mem alloc failed for *tensors");
    returnStatus = StatusCode::FAILURE;
    return returnStatus;
  }
  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    Qnn_Tensor_t wrapperTensor = tensorWrappers[tensorIdx];
    std::vector<size_t> dims;
    fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(wrapperTensor), QNN_TENSOR_GET_RANK(wrapperTensor));
    if (StatusCode::SUCCESS == returnStatus) {
      QNN_DEBUG("allocateBuffer successful");
      (*tensors)[tensorIdx] = QNN_TENSOR_INIT;
      returnStatus =
          (rwkv_app::deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor) == true
               ? StatusCode::SUCCESS
               : StatusCode::FAILURE);
    }
    if (StatusCode::SUCCESS == returnStatus) {
      QNN_DEBUG("deepCopyQnnTensorInfo successful");
      QNN_TENSOR_SET_MEM_TYPE(((*tensors) + tensorIdx), QNN_TENSORMEMTYPE_RAW);
    }
    Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;
    returnStatus = allocateBuffer(reinterpret_cast<uint8_t**>(&clientBuffer.data),
                                  dims,
                                  QNN_TENSOR_GET_DATA_TYPE((*tensors) + tensorIdx));
    datautil::StatusCode datautilStatus{datautil::StatusCode::SUCCESS};
    size_t length{0};
    std::tie(datautilStatus, length) =
        datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE((*tensors) + tensorIdx));
    if (datautilStatus != datautil::StatusCode::SUCCESS) {
      returnStatus = StatusCode::FAILURE;
    }
    clientBuffer.dataSize = length;
    QNN_TENSOR_SET_CLIENT_BUF(((*tensors) + tensorIdx), clientBuffer);
    if (StatusCode::SUCCESS != returnStatus) {
      QNN_ERROR("Failure in setupTensors, cleaning up resources");
      if (nullptr != (QNN_TENSOR_GET_CLIENT_BUF((*tensors) + tensorIdx)).data) {
        free(QNN_TENSOR_GET_CLIENT_BUF((*tensors) + tensorIdx).data);
      }
      tearDownTensors(*tensors, tensorIdx);
      *tensors     = nullptr;
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("Failure in setupTensors, done cleaning up resources");
      return returnStatus;
    }
  }
  return returnStatus;
}

// Setup details for all input and output tensors for graph execution.
iotensor::StatusCode iotensor::IOTensor::setupInputAndOutputTensors(
    Qnn_Tensor_t** inputs, Qnn_Tensor_t** outputs, qnn_wrapper_api::GraphInfo_t graphInfo) {
  auto returnStatus = StatusCode::SUCCESS;
  if (StatusCode::SUCCESS !=
      setupTensors(inputs, graphInfo.numInputTensors, (graphInfo.inputTensors))) {
    QNN_ERROR("Failure in setting up input tensors");
    returnStatus = StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS !=
      setupTensors(outputs, graphInfo.numOutputTensors, (graphInfo.outputTensors))) {
    QNN_ERROR("Failure in setting up output tensors");
    returnStatus = StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_ERROR("Failure in setupInputAndOutputTensors, cleaning up resources");
    if (nullptr != *inputs) {
      QNN_DEBUG("cleaning up input tensors");
      tearDownTensors(*inputs, graphInfo.numInputTensors);
      *inputs = nullptr;
    }
    if (nullptr != *outputs) {
      QNN_DEBUG("cleaning up output tensors");
      tearDownTensors(*outputs, graphInfo.numOutputTensors);
      *outputs = nullptr;
    }
    QNN_ERROR("Failure in setupInputAndOutputTensors, done cleaning up resources");
  }
  return returnStatus;
}

// Clean up all tensors related data after execution.
iotensor::StatusCode iotensor::IOTensor::tearDownTensors(Qnn_Tensor_t* tensors,
                                                         uint32_t tensorCount) {
  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    QNN_DEBUG("freeing resources for tensor: %d", tensorIdx);
    if (nullptr != QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx])) {
      QNN_DEBUG("freeing dimensions");
      free(QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx]));
    }
    if (nullptr != QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data) {
      QNN_DEBUG("freeing clientBuf.data");
      free(QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data);
    }
  }
  free(tensors);
  return StatusCode::SUCCESS;
}

// Clean up all input and output tensors after execution.
iotensor::StatusCode iotensor::IOTensor::tearDownInputAndOutputTensors(Qnn_Tensor_t* inputs,
                                                                       Qnn_Tensor_t* outputs,
                                                                       size_t numInputTensors,
                                                                       size_t numOutputTensors) {
  if (nullptr != inputs) {
    QNN_INFO("cleaning up resources for input tensors");
    tearDownTensors(inputs, numInputTensors);
    inputs = nullptr;
  }
  if (nullptr != outputs) {
    QNN_INFO("cleaning up resources for output tensors");
    tearDownTensors(outputs, numOutputTensors);
    outputs = nullptr;
  }
  return StatusCode::SUCCESS;
}

// Helper method to allocate a buffer.
iotensor::StatusCode iotensor::IOTensor::allocateBuffer(uint8_t** buffer,
                                                        std::vector<size_t> dims,
                                                        Qnn_DataType_t dataType) {
  size_t elementCount = datautil::calculateElementCount(dims);
  auto returnStatus   = StatusCode::SUCCESS;
  switch (dataType) {
    case QNN_DATATYPE_FLOAT_32:
      QNN_DEBUG("allocating float buffer");
      returnStatus = allocateBuffer<float>(reinterpret_cast<float**>(buffer), elementCount);
      break;

    case QNN_DATATYPE_FLOAT_16:
      QNN_DEBUG("allocating half buffer");
      returnStatus = allocateBuffer<half_float::half>(reinterpret_cast<half_float::half**>(buffer),
                                                      elementCount);
      break;

    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      QNN_DEBUG("allocating uint8_t buffer");
      returnStatus = allocateBuffer<uint8_t>(reinterpret_cast<uint8_t**>(buffer), elementCount);
      break;

    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      QNN_DEBUG("allocating uint16_t buffer");
      returnStatus = allocateBuffer<uint16_t>(reinterpret_cast<uint16_t**>(buffer), elementCount);
      break;

    case QNN_DATATYPE_UINT_32:
      QNN_DEBUG("allocating uint32_t buffer");
      returnStatus = allocateBuffer<uint32_t>(reinterpret_cast<uint32_t**>(buffer), elementCount);
      break;

    case QNN_DATATYPE_INT_8:
      QNN_DEBUG("allocating int8_t buffer");
      returnStatus = allocateBuffer<int8_t>(reinterpret_cast<int8_t**>(buffer), elementCount);
      break;

    case QNN_DATATYPE_INT_16:
      QNN_DEBUG("allocating int16_t buffer");
      returnStatus = allocateBuffer<int16_t>(reinterpret_cast<int16_t**>(buffer), elementCount);
      break;

    case QNN_DATATYPE_INT_32:
      QNN_DEBUG("allocating int32_t buffer");
      returnStatus = allocateBuffer<int32_t>(reinterpret_cast<int32_t**>(buffer), elementCount);
      break;

    case QNN_DATATYPE_BOOL_8:
      QNN_DEBUG("allocating bool buffer");
      returnStatus = allocateBuffer<uint8_t>(reinterpret_cast<uint8_t**>(buffer), elementCount);
      break;

    default:
      QNN_ERROR("Datatype %x not supported yet!", dataType);
      returnStatus = StatusCode::FAILURE;
      break;
  }
  return returnStatus;
}

// Helper method to allocate a buffer.
template <typename T>
iotensor::StatusCode iotensor::IOTensor::allocateBuffer(T** buffer, size_t& elementCount) {
  QNN_DEBUG("ElementCount: %d, sizeof(T): %d, total size: %d",
            elementCount,
            sizeof(T),
            elementCount * sizeof(T));
  *buffer = (T*)malloc(elementCount * sizeof(T));
  if (nullptr == *buffer) {
    QNN_ERROR("mem alloc failed for *buffer");
    return StatusCode::FAILURE;
  }
  return StatusCode::SUCCESS;
}

// Convert data to float or de-quantization. This is used when
// user requests for float output and the model produces
// non-float output.
iotensor::StatusCode iotensor::IOTensor::convertToFloat(float** out, Qnn_Tensor_t* tensor) {
  if (nullptr == tensor) {
    QNN_ERROR("tensors is nullptr");
    return StatusCode::FAILURE;
  }
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor));
  auto returnStatus   = StatusCode::SUCCESS;
  size_t elementCount = datautil::calculateElementCount(dims);
  returnStatus        = allocateBuffer<float>(out, elementCount);
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_ERROR("failure in allocateBuffer<float>");
    return returnStatus;
  }
  switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::tfNToFloat<uint8_t>(
              *out,
              reinterpret_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
              QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
              elementCount)) {
        QNN_ERROR("failure in tfNToFloat<uint8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UFIXED_POINT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::tfNToFloat<uint16_t>(
              *out,
              reinterpret_cast<uint16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
              QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
              elementCount)) {
        QNN_ERROR("failure in tfNToFloat<uint8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<uint8_t>(
              *out,
              reinterpret_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              elementCount)) {
        QNN_ERROR("failure in castToFloat<uint8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<uint16_t>(
              *out,
              reinterpret_cast<uint16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              elementCount)) {
        QNN_ERROR("failure in castToFloat<uint16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<uint32_t>(
              *out,
              reinterpret_cast<uint32_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              elementCount)) {
        QNN_ERROR("failure in castToFloat<uint32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<int8_t>(
              *out,
              reinterpret_cast<int8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              elementCount)) {
        QNN_ERROR("failure in castToFloat<int8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<int16_t>(
              *out,
              reinterpret_cast<int16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              elementCount)) {
        QNN_ERROR("failure in castToFloat<int16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<int32_t>(
              *out,
              reinterpret_cast<int32_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              elementCount)) {
        QNN_ERROR("failure in castToFloat<int32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_BOOL_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<uint8_t>(
              *out,
              reinterpret_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
              elementCount)) {
        QNN_ERROR("failure in castToFloat<bool>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    default:
      QNN_ERROR("Datatype not supported yet!");
      returnStatus = StatusCode::FAILURE;
      break;
  }
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_DEBUG("freeing *out");
    if (*out != nullptr) {
      free(*out);
      *out = nullptr;
    }
  }
  return returnStatus;
}

// Helper method to convert Output tensors to float and write them
// out to files.
iotensor::StatusCode iotensor::IOTensor::convertAndWriteOutputTensorInFloat(
    Qnn_Tensor_t* output,
    std::vector<std::string> outputPaths,
    std::string fileName,
    size_t outputBatchSize) {
  if (nullptr == output) {
    QNN_ERROR("output is nullptr");
    return StatusCode::FAILURE;
  }

  auto returnStatus = StatusCode::SUCCESS;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output), QNN_TENSOR_GET_RANK(output));
  float* floatBuffer = nullptr;
  returnStatus       = convertToFloat(&floatBuffer, output);
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_ERROR("failure in convertToFloat");
    return StatusCode::FAILURE;
  }
  uint8_t* bufferToWrite = reinterpret_cast<uint8_t*>(floatBuffer);
  if (datautil::StatusCode::SUCCESS !=
      datautil::writeBatchDataToFile(
          outputPaths, fileName, dims, QNN_DATATYPE_FLOAT_32, bufferToWrite, outputBatchSize)) {
    QNN_ERROR("failure in writeBatchDataToFile");
    returnStatus = StatusCode::FAILURE;
  }
  if (nullptr != floatBuffer) {
    QNN_DEBUG("freeing floatBuffer");
    free(floatBuffer);
    floatBuffer = nullptr;
  }
  return returnStatus;
}

// Helper method to write out output. There is no de-quantization here.
// Just write output as is to files.
iotensor::StatusCode iotensor::IOTensor::writeOutputTensor(Qnn_Tensor_t* output,
                                                           std::vector<std::string> outputPaths,
                                                           std::string fileName,
                                                           size_t outputBatchSize) {
  if (nullptr == output) {
    QNN_ERROR("output is nullptr");
    return StatusCode::FAILURE;
  }
  auto returnStatus = StatusCode::SUCCESS;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output), QNN_TENSOR_GET_RANK(output));
  uint8_t* bufferToWrite = reinterpret_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(output).data);
  if (datautil::StatusCode::SUCCESS !=
      datautil::writeBatchDataToFile(outputPaths,
                                     fileName,
                                     dims,
                                     QNN_TENSOR_GET_DATA_TYPE(output),
                                     bufferToWrite,
                                     outputBatchSize)) {
    QNN_ERROR("failure in writeBatchDataToFile");
    returnStatus = StatusCode::FAILURE;
  }
  return returnStatus;
}

// Write out all output tensors to files. If output_data_type is float,
// then all outputs will be raw floats regardless of what the model outputs.
// If the output_data_type is native, then output is written as produced by the model.
// Also, for native option, a json with quantization parameters is written out.
// If output_data_type is float_and_native, both above are done.
// If the output in the graph is float, then output_data_type has no effect.
iotensor::StatusCode iotensor::IOTensor::writeOutputTensors(uint32_t graphIdx,
                                                            size_t startIdx,
                                                            char* graphName,
                                                            Qnn_Tensor_t* outputs,
                                                            uint32_t numOutputs,
                                                            iotensor::OutputDataType outputDatatype,
                                                            uint32_t graphsCount,
                                                            std::string outputPath,
                                                            size_t numInputFilesPopulated,
                                                            size_t outputBatchSize) {
  if (nullptr == outputs) {
    QNN_ERROR("Received nullptr");
    return StatusCode::FAILURE;
  }
  if (graphsCount > 1) {
    if (nullptr != graphName && strlen(graphName) > 0) {
      outputPath += (pal::Path::getSeparator() + std::string(graphName));
    } else {
      outputPath += (pal::Path::getSeparator() + std::string("Graph_") + std::to_string(graphIdx));
    }
  }
  auto returnStatus = StatusCode::SUCCESS;
  std::vector<std::string> outputPaths;
  for (size_t idx = 0; idx < numInputFilesPopulated; idx++) {
    std::string output = outputPath + (pal::Path::getSeparator() + std::string("Result_") +
                                       std::to_string(startIdx + idx));
    outputPaths.push_back(output);
  }
  for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
    QNN_DEBUG("Writing output for outputIdx: %d", outputIdx);
    std::string outputFilePrefix;
    if (nullptr != QNN_TENSOR_GET_NAME(outputs[outputIdx]) &&
        strlen(QNN_TENSOR_GET_NAME(outputs[outputIdx])) > 0) {
      outputFilePrefix = std::string(QNN_TENSOR_GET_NAME(outputs[outputIdx]));
    } else {
      outputFilePrefix = std::string("Output_") + std::to_string(outputIdx);
    }
    auto outputFile       = outputFilePrefix + std::string(".raw");
    auto outputFileNative = outputFilePrefix + std::string("_native.raw");
    if (QNN_TENSOR_GET_DATA_TYPE(outputs[outputIdx]) == QNN_DATATYPE_FLOAT_32) {
      QNN_DEBUG("Writing in output->dataType == QNN_DATATYPE_FLOAT_32");
      returnStatus =
          writeOutputTensor(&(outputs[outputIdx]), outputPaths, outputFile, outputBatchSize);
    } else if (outputDatatype == OutputDataType::FLOAT_ONLY) {
      QNN_DEBUG("Writing in output->dataType == OutputDataType::FLOAT_ONLY");
      returnStatus = convertAndWriteOutputTensorInFloat(
          &(outputs[outputIdx]), outputPaths, outputFile, outputBatchSize);
    } else if (outputDatatype == OutputDataType::NATIVE_ONLY) {
      QNN_DEBUG("Writing in output->dataType == OutputDataType::NATIVE_ONLY");
      returnStatus =
          writeOutputTensor(&(outputs[outputIdx]), outputPaths, outputFileNative, outputBatchSize);
    } else if (outputDatatype == OutputDataType::FLOAT_AND_NATIVE) {
      QNN_DEBUG("Writing in output->dataType == OutputDataType::FLOAT_AND_NATIVE");
      returnStatus = convertAndWriteOutputTensorInFloat(
          &(outputs[outputIdx]), outputPaths, outputFile, outputBatchSize);
      if (StatusCode::SUCCESS == returnStatus) {
        returnStatus = writeOutputTensor(
            &(outputs[outputIdx]), outputPaths, outputFileNative, outputBatchSize);
      }
    }
  }
  return returnStatus;
}

// Helper method to allocate a buffer and copy data to it.
iotensor::StatusCode iotensor::IOTensor::allocateAndCopyBuffer(uint8_t** buffer,
                                                               Qnn_Tensor_t* tensor) {
  if (nullptr == tensor) {
    return StatusCode::FAILURE;
  }
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor));
  datautil::StatusCode datautilStatus;
  size_t length;
  std::tie(datautilStatus, length) =
      datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE(tensor));
  if (datautilStatus != datautil::StatusCode::SUCCESS) {
    return StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS != allocateBuffer(buffer, dims, QNN_TENSOR_GET_DATA_TYPE(tensor))) {
    QNN_ERROR("failure in allocateBuffer");
    return StatusCode::FAILURE;
  }
  pal::StringOp::memscpy(*buffer,
                         length * sizeof(uint8_t),
                         QNN_TENSOR_GET_CLIENT_BUF(tensor).data,
                         length * sizeof(uint8_t));
  return StatusCode::SUCCESS;
}

iotensor::StatusCode iotensor::IOTensor::fillDims(std::vector<size_t>& dims,
                                                  uint32_t* inDimensions,
                                                  uint32_t rank) {
  if (nullptr == inDimensions) {
    QNN_ERROR("input dimensions is nullptr");
    return StatusCode::FAILURE;
  }
  for (size_t r = 0; r < rank; r++) {
    dims.push_back(inDimensions[r]);
  }
  return StatusCode::SUCCESS;
}

iotensor::OutputDataType iotensor::parseOutputDataType(std::string dataTypeString) {
  std::transform(dataTypeString.begin(), dataTypeString.end(), dataTypeString.begin(), ::tolower);
  OutputDataType parsedDataType = OutputDataType::INVALID;
  if (dataTypeString == "float_only") {
    parsedDataType = OutputDataType::FLOAT_ONLY;
  } else if (dataTypeString == "native_only") {
    parsedDataType = OutputDataType::NATIVE_ONLY;
  } else if (dataTypeString == "float_and_native") {
    parsedDataType = OutputDataType::FLOAT_AND_NATIVE;
  }
  return parsedDataType;
}

iotensor::InputDataType iotensor::parseInputDataType(std::string dataTypeString) {
  std::transform(dataTypeString.begin(), dataTypeString.end(), dataTypeString.begin(), ::tolower);
  InputDataType parsedDataType = InputDataType::INVALID;
  if (dataTypeString == "float") {
    parsedDataType = InputDataType::FLOAT;
  } else if (dataTypeString == "native") {
    parsedDataType = InputDataType::NATIVE;
  }
  return parsedDataType;
}