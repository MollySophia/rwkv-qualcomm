//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================
#include <iostream>
#include <string>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace wkv7_state {

Qnn_ErrorHandle_t execute(CustomOp* operation) {
  /*
   * To have good performance and stability, it is required to avoid heap memory
   * allocation in this function. The heap memory allocation includes but not
   * limited to calling malloc, operator new, constructing STL container objects
   * like std::vector with default allocator, and adding items like calling
   * std::vector::push_back to STL container objects with default allocator.
   *
   * Please check in SDK documentation for more information.
   */

  float* r = (float*)operation->getInput(0)->data;
  float* w = (float*)operation->getInput(1)->data;
  float* k = (float*)operation->getInput(2)->data;
  float* v = (float*)operation->getInput(3)->data;
  float* a = (float*)operation->getInput(4)->data;
  float* b = (float*)operation->getInput(5)->data;
  float* state_in = (float*)operation->getInput(6)->data;
  float* output = (float*)operation->getOutput(0)->data;
  float* state_out = (float*)operation->getOutput(1)->data;

  int num_heads = operation->getInput(6)->currentDimensions[0];
  int head_size = operation->getInput(6)->currentDimensions[1];
  // int seq_length = operation->getInput(0)->currentDimensions[0];
  int seq_length = operation->getInput(0)->currentDimensions[0] / num_heads;

  for (int t = 0; t < seq_length; t++) {
    if (t > 0) state_in = state_out;
    for (int h = 0; h < num_heads; h++) {
      for (int i = 0; i < head_size; i++) {
        auto v_val = v[t * num_heads * head_size + h * head_size + i];

        float sa = 0, result = 0;
        for (int j = 0; j < head_size; j++) {
          sa += a[t * num_heads * head_size + h * head_size + j] * state_in[h * head_size * head_size + i * head_size + j];
        }

        for (int j = 0; j < head_size; j++) {
          auto r_val = r[t * num_heads * head_size + h * head_size + j];
          auto w_val = w[t * num_heads * head_size + h * head_size + j];
          auto k_val = k[t * num_heads * head_size + h * head_size + j];
          auto b_val = b[t * num_heads * head_size + h * head_size + j];
          auto kv_val = k_val * v_val;
          auto state_val = state_in[h * head_size * head_size + i * head_size + j] * w_val + kv_val + sa * b_val;
          result += state_val * r_val;
          state_out[h * head_size * head_size + i * head_size + j] = state_val;
        }
        output[t * num_heads * head_size + h * head_size + i] = result;
      }
    }
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 6, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) {

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }


  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "wkv7_state"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 6, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace wkv7_state

CustomOpRegistration_t* register_Wkv7StateCustomOp() {
  using namespace wkv7_state;
  static CustomOpRegistration_t WkvRegister = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &WkvRegister;
}

REGISTER_OP(wkv7_state, register_Wkv7StateCustomOp);
