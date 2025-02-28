//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnTypeDef.hpp"

/**
 * @brief Frees all memory allocated tensor attributes.
 *
 * @param[in] tensor Qnn_Tensor_t object to free
 *
 * @return Error code
 */
ModelError_t freeQnnTensor(Qnn_Tensor_t &tensor);

/**
 * @brief Loops through and frees all memory allocated tensor attributes for each tensor
 * object.
 *
 * @param[in] tensors array of tensor objects to free
 *
 * @param[in] numTensors length of the above tensors array
 *
 * @return Error code
 */
ModelError_t freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors);

/**
 * @brief A helper function to free memory malloced for communicating the Graph for a model(s)
 *
 * @param[in] graphsInfo Pointer pointing to location of graph objects
 *
 * @param[in] numGraphs The number of graph objects the above pointer is pointing to
 *
 * @return Error code
 *
 */
ModelError_t freeGraphsInfo(GraphInfoPtr_t **graphsInfo, uint32_t numGraphs);
