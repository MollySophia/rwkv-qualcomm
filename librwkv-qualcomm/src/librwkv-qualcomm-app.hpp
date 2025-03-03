#pragma once

#include <memory>
#include <queue>
#include <vector>

#include "IOTensor.hpp"
#include "Interfaces.hpp"
#include "half.hpp"

namespace qnn {
namespace tools {
namespace rwkv_app {

enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAILURE_INPUT_LIST_EXHAUSTED,
  FAILURE_SYSTEM_ERROR,
  FAILURE_SYSTEM_COMMUNICATION_ERROR,
  QNN_FEATURE_UNSUPPORTED
};

const int max_chunks = 8;

class QnnRwkvApp {
 public:
  QnnRwkvApp(QnnFunctionPointers qnnFunctionPointers,
               void *backendHandle,
               void *modelHandle,
               std::vector<std::vector<float>> embedding = {},
               std::string cachedBinaryPath            = "",
               std::string saveBinaryName              = "");

  StatusCode initialize();

  StatusCode initializeBackend();

  StatusCode createContext();

  StatusCode composeGraphs();

  StatusCode finalizeGraphs();

  StatusCode createPowerConfigId();

  StatusCode setPowerConfig();

  StatusCode destroyPowerConfigId();

  StatusCode setRpcLatencyAndPolling();

  StatusCode initializeTensors();

  StatusCode execute(int token);

  StatusCode executeSequence(std::vector<int> &tokens);

  StatusCode registerOpPackages();

  StatusCode createFromBinary(uint8_t *binary, size_t binarySize);

  StatusCode saveBinary();

  StatusCode freeContext();

  StatusCode terminateBackend();

  StatusCode freeGraphs();

  Qnn_ContextHandle_t getContext();

  std::string getBackendBuildId();

  StatusCode isDevicePropertySupported();

  StatusCode createDevice();

  size_t getQnnDatatypeSize(Qnn_DataType_t dataType);

  StatusCode freeDevice();

  StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);

  void fillQuantizedTensor(float value, Qnn_Tensor_t *tensor);

  virtual ~QnnRwkvApp();

  std::vector<half_float::half> m_lastOutput;

  uint32_t powerConfigId;
  uint32_t deviceId = 0;
  uint32_t coreId = 0;

  QnnFunctionPointers m_qnnFunctionPointers;
  std::string m_outputPath;
  std::string m_saveBinaryName;
  std::string m_cachedBinaryPath;
  std::vector<std::string> m_opPackagePaths;
  uint8_t *m_binaryBuffer = nullptr;
  uint64_t m_binarySize = 0;
  QnnBackend_Config_t **m_backendConfig = nullptr;
  Qnn_ContextHandle_t m_context[max_chunks] = {nullptr};
  QnnContext_Config_t **m_contextConfig = nullptr;
  GraphInfo_t **m_decodeGraphsInfo;
  GraphInfo_t **m_prefillGraphsInfo;
  uint32_t m_decodeGraphsCount;
  uint32_t m_prefillGraphsCount;
  void *m_backendLibraryHandle;
  void *m_modelHandle;
  IOTensor *m_ioTensor;
  Qnn_Tensor_t *m_inputTensors[max_chunks] = {nullptr};
  Qnn_Tensor_t *m_outputTensors[max_chunks] = {nullptr};
  Qnn_Tensor_t *m_prefillInputTensors[max_chunks] = {nullptr};
  Qnn_Tensor_t *m_prefillOutputTensors[max_chunks] = {nullptr};
  std::vector<std::vector<float>> m_embedding = {};
  bool m_tensorsInitialized = false;
  bool m_isBackendInitialized;
  bool m_isContextCreated;

  GraphConfigInfo_t **m_graphConfigsInfo = nullptr;
  uint32_t m_graphConfigsInfoCount;
  Qnn_LogHandle_t m_logHandle         = nullptr;
  Qnn_BackendHandle_t m_backendHandle = nullptr;
  Qnn_DeviceHandle_t m_deviceHandle   = nullptr;

  Qnn_Tensor_t *m_logitsOutputTensor = nullptr;

  std::vector<std::unordered_map<std::string, void*>> m_decodeGraphsTensorNameToTensorPointer;
  std::vector<std::unordered_map<std::string, size_t>> m_decodeGraphsTensorNameToSize;
  std::vector<std::unordered_map<std::string, void*>> m_prefillGraphsTensorNameToTensorPointer;
  std::vector<std::unordered_map<std::string, size_t>> m_prefillGraphsTensorNameToSize;

  int m_prefillSequenceLength = 0;

  std::chrono::duration<double> m_lastInferenceTime;
};
}  // namespace rwkv_app
}  // namespace tools
}  // namespace qnn
