//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================
#include <iostream>
#include <string>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace wkv6 {

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

  float* k = (float*)operation->getInput(0)->data;
  float* v = (float*)operation->getInput(1)->data;
  float* r = (float*)operation->getInput(2)->data;
  float* state_in = (float*)operation->getInput(3)->data;
  float* tf = (float*)operation->getInput(4)->data;
  float* td = (float*)operation->getInput(5)->data;
  float* output = (float*)operation->getOutput(0)->data;
  float* state_out = (float*)operation->getOutput(1)->data;

  int seq_length = operation->getInput(0)->currentDimensions[0];
  int num_heads = operation->getInput(3)->currentDimensions[0];
  int head_size = operation->getInput(3)->currentDimensions[1];
  memset(output, 0, seq_length * num_heads * head_size * sizeof(float));
  for (int t = 0; t < seq_length; t++) {
    if (t > 0) state_in = state_out;
    for (int h = 0; h < num_heads; h++) {
      for (int i = 0; i < head_size; i++) {
        auto k_val = k[t * num_heads * head_size + h * head_size + i];
        auto r_val = r[t * num_heads * head_size + h * head_size + i];
        auto td_val = td[t * num_heads * head_size + h * head_size + i];
        auto tf_val = tf[h * head_size + i];
        for (int j = 0; j < head_size; j++) {
          auto v_val = v[t * num_heads * head_size + h * head_size + j];
          auto kv_val = k_val * v_val;
          auto prev_state_val = state_in[h * head_size * head_size + i * head_size + j];
          output[t * num_heads * head_size + h * head_size + j] += r_val * (kv_val * tf_val + prev_state_val);
          state_out[h * head_size * head_size + i * head_size + j] = prev_state_val * td_val + kv_val;
        }
      }
    }
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 6, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 2, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

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
      strcmp(opConfig.v1.typeName, "wkv6"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 6, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 2, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace wkv6

CustomOpRegistration_t* register_WkvCustomOp() {
  using namespace wkv6;
  static CustomOpRegistration_t WkvRegister = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &WkvRegister;
}

REGISTER_OP(wkv6, register_WkvCustomOp);
