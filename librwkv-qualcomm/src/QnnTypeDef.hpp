//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QNN_TYPE_DEF_H_
#define QNN_TYPE_DEF_H_

#include "Logger.hpp"
#include "QnnInterface.h"
#include "QnnTypeMacros.hpp"
#include "QnnTypes.h"

typedef enum ModelError {
  MODEL_NO_ERROR               = 0,
  MODEL_TENSOR_ERROR           = 1,
  MODEL_PARAMS_ERROR           = 2,
  MODEL_NODES_ERROR            = 3,
  MODEL_GRAPH_ERROR            = 4,
  MODEL_CONTEXT_ERROR          = 5,
  MODEL_GENERATION_ERROR       = 6,
  MODEL_SETUP_ERROR            = 7,
  MODEL_INVALID_ARGUMENT_ERROR = 8,
  MODEL_FILE_ERROR             = 9,
  MODEL_MEMORY_ALLOCATE_ERROR  = 10,
  // Value selected to ensure 32 bits.
  MODEL_UNKNOWN_ERROR = 0x7FFFFFFF
} ModelError_t;

using TensorWrapper = Qnn_Tensor_t;
#define GET_TENSOR_WRAPPER_TENSOR(tensorWrapper) tensorWrapper
#define GET_TENSOR_WRAPPER_NAME(tensorWrapper)   QNN_TENSOR_GET_NAME(tensorWrapper)

typedef struct GraphInfo {
  Qnn_GraphHandle_t graph;
  char* graphName;
  TensorWrapper* inputTensors;
  uint32_t numInputTensors;
  TensorWrapper* outputTensors;
  uint32_t numOutputTensors;
} GraphInfo_t;
typedef GraphInfo_t* GraphInfoPtr_t;

typedef struct GraphConfigInfo {
  char* graphName;
  const QnnGraph_Config_t** graphConfigs;
} GraphConfigInfo_t;

#endif  // QNN_TYPE_DEF_H_
