#pragma once

#include <memory>
#include <queue>

#include "IOTensor.hpp"
#include "Interfaces.hpp"
#include "tokenizer.h"
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

class QnnRwkvApp {
 public:
  QnnRwkvApp(QnnFunctionPointers qnnFunctionPointers,
               std::string promptPath,
               std::string configPath,
               std::string tokenizerPath,
               void *backendHandle,
               ProfilingLevel profilingLevel           = ProfilingLevel::OFF,
               std::string cachedBinaryPath            = "",
               std::string saveBinaryName              = "");

  // @brief Print a message to STDERR then return a nonzero
  //  exit status.
  int32_t reportError(const std::string &err);

  StatusCode initialize();

  StatusCode initializeBackend();

  StatusCode createContext();

  StatusCode composeGraphs();

  StatusCode finalizeGraphs();

  StatusCode createPowerConfigId();

  StatusCode setPowerConfig();

  StatusCode destroyPowerConfigId();

  StatusCode setRpcLatencyAndPolling();

  StatusCode execute(int token);

  // StatusCode registerOpPackages();

  StatusCode createFromBinary();

  StatusCode saveBinary();

  StatusCode freeContext();

  StatusCode terminateBackend();

  StatusCode freeGraphs();

  Qnn_ContextHandle_t getContext();

  StatusCode initializeProfiling();

  std::string getBackendBuildId();

  StatusCode isDevicePropertySupported();

  StatusCode createDevice();

  StatusCode freeDevice();

  StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);

  virtual ~QnnRwkvApp();

  std::shared_ptr<TokenizerBase> m_tokenizer;
  // std::vector<std::vector<float>> m_stateTensors;
  std::vector<half_float::half> m_lastOutput;

  int m_headSize;
  int m_nEmbd;
  int m_numLayer;
  int m_nATT;
  int m_nFFN;
  int m_vocabSize;

 private:
  StatusCode extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);

  StatusCode extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);

  StatusCode extractProfilingEvent(QnnProfile_EventId_t profileEventId);

  uint32_t powerConfigId;
  uint32_t deviceId = 0;
  uint32_t coreId = 0;

  QnnFunctionPointers m_qnnFunctionPointers;
  std::string m_promptPath;
  std::string m_configPath;
  std::string m_tokenizerPath;
  std::vector<std::unordered_map<std::string, uint32_t>> m_inputNameToIndex;
  std::string m_outputPath;
  std::string m_saveBinaryName;
  std::string m_cachedBinaryPath;
  QnnBackend_Config_t **m_backendConfig = nullptr;
  Qnn_ContextHandle_t m_context         = nullptr;
  QnnContext_Config_t **m_contextConfig = nullptr;
  ProfilingLevel m_profilingLevel;
  qnn_wrapper_api::GraphInfo_t **m_graphsInfo;
  uint32_t m_graphsCount;
  void *m_backendLibraryHandle;
  iotensor::IOTensor m_ioTensor;
  Qnn_Tensor_t *m_inputTensors = nullptr;
  Qnn_Tensor_t *m_outputTensors = nullptr;
  bool m_isBackendInitialized;
  bool m_isContextCreated;
  Qnn_ProfileHandle_t m_profileBackendHandle              = nullptr;
  qnn_wrapper_api::GraphConfigInfo_t **m_graphConfigsInfo = nullptr;
  uint32_t m_graphConfigsInfoCount;
  Qnn_LogHandle_t m_logHandle         = nullptr;
  Qnn_BackendHandle_t m_backendHandle = nullptr;
  Qnn_DeviceHandle_t m_deviceHandle   = nullptr;
};
}  // namespace rwkv_app
}  // namespace tools
}  // namespace qnn
