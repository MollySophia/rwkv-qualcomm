#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "Logger.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#include "PAL/StringOp.hpp"
#include "Utils.hpp"
#include "QnnTypeMacros.hpp"

using namespace qnn;
using namespace qnn::tools;

void rwkv_app::split(std::vector<std::string> &splitString,
                       const std::string &tokenizedString,
                       const char separator) {
  splitString.clear();
  std::istringstream tokenizedStringStream(tokenizedString);
  while (!tokenizedStringStream.eof()) {
    std::string value;
    getline(tokenizedStringStream, value, separator);
    if (!value.empty()) {
      splitString.push_back(value);
    }
  }
}

rwkv_app::ProfilingLevel rwkv_app::parseProfilingLevel(std::string profilingLevelString) {
  std::transform(profilingLevelString.begin(),
                 profilingLevelString.end(),
                 profilingLevelString.begin(),
                 ::tolower);
  ProfilingLevel parsedProfilingLevel = ProfilingLevel::INVALID;
  if (profilingLevelString == "off") {
    parsedProfilingLevel = ProfilingLevel::OFF;
  } else if (profilingLevelString == "basic") {
    parsedProfilingLevel = ProfilingLevel::BASIC;
  } else if (profilingLevelString == "detailed") {
    parsedProfilingLevel = ProfilingLevel::DETAILED;
  }
  return parsedProfilingLevel;
}

bool rwkv_app::deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
  if (nullptr == dst || nullptr == src) {
    QNN_ERROR("Received nullptr");
    return false;
  }
  // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
  // to correctly assign values
  dst->version           = src->version;
  const char *tensorName = QNN_TENSOR_GET_NAME(src);
  if (!tensorName) {
    QNN_TENSOR_SET_NAME(dst, nullptr);
  } else {
    QNN_TENSOR_SET_NAME(dst, pal::StringOp::strndup(tensorName, strlen(tensorName)));
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
  } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
    if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
          QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets *
          sizeof(Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0;
             idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
             idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
        }
      }
    }
  }
  QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
  QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
  QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
  if (QNN_TENSOR_GET_RANK(src) > 0) {
    QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
    if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
      pal::StringOp::memscpy(QNN_TENSOR_GET_DIMENSIONS(dst),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t),
                             QNN_TENSOR_GET_DIMENSIONS(src),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
    }
  }
  return true;
}

bool rwkv_app::copyTensorsInfo(const Qnn_Tensor_t *tensorsInfoSrc,
                                 Qnn_Tensor_t *&tensorWrappers,
                                 uint32_t tensorsCount) {
  QNN_FUNCTION_ENTRY_LOG;
  auto returnStatus = true;
  tensorWrappers    = (Qnn_Tensor_t *)calloc(tensorsCount, sizeof(Qnn_Tensor_t));
  if (nullptr == tensorWrappers) {
    QNN_ERROR("Failed to allocate memory for tensorWrappers.");
    return false;
  }
  if (returnStatus) {
    for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
      QNN_DEBUG("Extracting tensorInfo for tensor Idx: %d", tIdx);
      tensorWrappers[tIdx] = QNN_TENSOR_INIT;
      deepCopyQnnTensorInfo(&tensorWrappers[tIdx], &tensorsInfoSrc[tIdx]);
    }
  }
  QNN_FUNCTION_EXIT_LOG;
  return returnStatus;
}

bool rwkv_app::copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t *graphInfoSrc,
                                  qnn_wrapper_api::GraphInfo_t *graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) {
    graphInfoDst->graphName =
        pal::StringOp::strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName));
  }
  graphInfoDst->inputTensors    = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors    = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                         graphInfoDst->outputTensors,
                         graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

bool rwkv_app::copyGraphsInfo(const QnnSystemContext_GraphInfo_t *graphsInput,
                                const uint32_t numGraphs,
                                qnn_wrapper_api::GraphInfo_t **&graphsInfo) {
  QNN_FUNCTION_ENTRY_LOG;
  if (!graphsInput) {
    QNN_ERROR("Received nullptr for graphsInput.");
    return false;
  }
  auto returnStatus = true;
  graphsInfo =
      (qnn_wrapper_api::GraphInfo_t **)calloc(numGraphs, sizeof(qnn_wrapper_api::GraphInfo_t *));
  qnn_wrapper_api::GraphInfo_t *graphInfoArr =
      (qnn_wrapper_api::GraphInfo_t *)calloc(numGraphs, sizeof(qnn_wrapper_api::GraphInfo_t));
  if (nullptr == graphsInfo || nullptr == graphInfoArr) {
    QNN_ERROR("Failure to allocate memory for *graphInfo");
    returnStatus = false;
  }
  if (true == returnStatus) {
    for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
      QNN_DEBUG("Extracting graphsInfo for graph Idx: %d", gIdx);
      if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        copyGraphsInfoV1(&graphsInput[gIdx].graphInfoV1, &graphInfoArr[gIdx]);
      }
      graphsInfo[gIdx] = graphInfoArr + gIdx;
    }
  }
  if (true != returnStatus) {
    QNN_ERROR("Received an ERROR during extractGraphsInfo. Freeing resources.");
    if (graphsInfo) {
      for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
        if (graphsInfo[gIdx]) {
          if (nullptr != graphsInfo[gIdx]->graphName) {
            free(graphsInfo[gIdx]->graphName);
            graphsInfo[gIdx]->graphName = nullptr;
          }
          qnn_wrapper_api::freeQnnTensors(graphsInfo[gIdx]->inputTensors,
                                          graphsInfo[gIdx]->numInputTensors);
          qnn_wrapper_api::freeQnnTensors(graphsInfo[gIdx]->outputTensors,
                                          graphsInfo[gIdx]->numOutputTensors);
        }
      }
      free(*graphsInfo);
    }
    free(graphsInfo);
    graphsInfo = nullptr;
  }
  QNN_FUNCTION_EXIT_LOG;
  return true;
}

bool rwkv_app::copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binaryInfo,
                                          qnn_wrapper_api::GraphInfo_t **&graphsInfo,
                                          uint32_t &graphsCount) {
  if (nullptr == binaryInfo) {
    QNN_ERROR("binaryInfo is nullptr.");
    return false;
  }
  graphsCount = 0;
  if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    if (binaryInfo->contextBinaryInfoV1.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV1.graphs,
                          binaryInfo->contextBinaryInfoV1.numGraphs,
                          graphsInfo)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV1.numGraphs;
      return true;
    }
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    if (binaryInfo->contextBinaryInfoV2.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV2.graphs,
                          binaryInfo->contextBinaryInfoV2.numGraphs,
                          graphsInfo)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV2.numGraphs;
      return true;
    }
  }
  QNN_ERROR("Unrecognized system context binary info version.");
  return false;
}

QnnLog_Level_t rwkv_app::parseLogLevel(std::string logLevelString) {
  QNN_FUNCTION_ENTRY_LOG;
  std::transform(logLevelString.begin(), logLevelString.end(), logLevelString.begin(), ::tolower);
  QnnLog_Level_t parsedLogLevel = QNN_LOG_LEVEL_MAX;
  if (logLevelString == "error") {
    parsedLogLevel = QNN_LOG_LEVEL_ERROR;
  } else if (logLevelString == "warn") {
    parsedLogLevel = QNN_LOG_LEVEL_WARN;
  } else if (logLevelString == "info") {
    parsedLogLevel = QNN_LOG_LEVEL_INFO;
  } else if (logLevelString == "verbose") {
    parsedLogLevel = QNN_LOG_LEVEL_VERBOSE;
  } else if (logLevelString == "debug") {
    parsedLogLevel = QNN_LOG_LEVEL_DEBUG;
  }
  QNN_FUNCTION_EXIT_LOG;
  return parsedLogLevel;
}
