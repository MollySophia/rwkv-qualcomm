#pragma once

#include <memory>
#include <queue>
#include <vector>

#include "IOTensor.hpp"
#include "Interfaces.hpp"
#include "half.hpp"
#include "rmpack.h"

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
               std::string cachedBinaryPath            = "",
               std::string saveBinaryName              = "");

  ~QnnRwkvApp();

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

  StatusCode createFromBinary(int spill_fill_buffer_size = 0);

  StatusCode createFromBinaryListAsync();

  StatusCode parseGraphsInfo(std::vector<GraphInfo_t **> &graphInfos, std::vector<uint32_t> &graphCounts);

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

  std::vector<half_float::half> m_lastOutput;

  uint32_t powerConfigId;
  uint32_t deviceId = 0;
  uint32_t coreId = 0;

  QnnFunctionPointers m_qnnFunctionPointers;
  std::string m_outputPath;
  std::string m_saveBinaryName;
  std::string m_cachedBinaryPath;
  std::vector<std::string> m_opPackagePaths;
  QnnBackend_Config_t **m_backendConfig = nullptr;
  Qnn_ContextHandle_t m_context[max_chunks] = {nullptr};
  QnnContext_Config_t **m_contextConfig = nullptr;
  int n_chunks = 0;
  GraphInfo_t **m_decodeGraphsInfo;
  GraphInfo_t **m_prefillGraphsInfo;
  uint32_t m_decodeGraphsCount;
  uint32_t m_prefillGraphsCount;

  std::vector<GraphInfo_t **> m_graphInfos;
  std::vector<uint32_t> m_graphCounts;

  void *m_backendLibraryHandle;
  void *m_modelHandle;
  IOTensor *m_ioTensor;
  Qnn_Tensor_t *m_inputTensors[max_chunks] = {nullptr};
  Qnn_Tensor_t *m_outputTensors[max_chunks] = {nullptr};
  Qnn_Tensor_t *m_prefillInputTensors[max_chunks] = {nullptr};
  Qnn_Tensor_t *m_prefillOutputTensors[max_chunks] = {nullptr};
  std::shared_ptr<uint8_t> m_embedding = nullptr;
  std::shared_ptr<uint8_t> m_lmhead_weight = nullptr;
  bool m_tensorsInitialized = false;
  bool m_isBackendInitialized;
  bool m_isContextCreated;

  RMPack *m_rmpack = nullptr;

  int m_hidden_size = 0;
  int m_vocab_size = 0;

  std::mutex m_updateCallBackMutex;

  void updateContext(Qnn_ContextHandle_t context, uint32_t contextId) {
    std::lock_guard<std::mutex> lock(m_updateCallBackMutex);
    m_context[contextId] = context;
  }

  void updateQnnApiGraphsandContextsInfo(std::string graphName,
                                         Qnn_GraphHandle_t graph,
                                         uint32_t contextId) {
    // set graph handle to GraphInfo
    std::lock_guard<std::mutex> lock(m_updateCallBackMutex);
    for (int i = 0; i < m_graphCounts[contextId]; i++) {
      if (std::string(m_graphInfos[contextId][i]->graphName) == graphName) {
        m_graphInfos[contextId][i]->graph = graph;
        break;
      }
    }
  }

  static void contextNotifyFn(Qnn_ContextHandle_t context,
    Qnn_GraphHandle_t graph,
    const char* graph_name,
    QnnContext_createFromBinaryAsyncNotifyType_t completeType,
    void* notifyParam,
    Qnn_ErrorHandle_t status
  );

  GraphConfigInfo_t **m_graphConfigsInfo = nullptr;
  uint32_t m_graphConfigsInfoCount;
  Qnn_LogHandle_t m_logHandle         = nullptr;
  Qnn_BackendHandle_t m_backendHandle = nullptr;
  Qnn_DeviceHandle_t m_deviceHandle   = nullptr;

  Qnn_Tensor_t *m_logitsOutputTensor = nullptr;
  std::vector<float> m_logitsOutput;

  std::vector<std::unordered_map<std::string, void*>> m_decodeGraphsTensorNameToTensorPointer;
  std::vector<std::unordered_map<std::string, size_t>> m_decodeGraphsTensorNameToSize;
  std::vector<std::unordered_map<std::string, void*>> m_prefillGraphsTensorNameToTensorPointer;
  std::vector<std::unordered_map<std::string, size_t>> m_prefillGraphsTensorNameToSize;

  int m_prefillSequenceLength = 0;

  std::chrono::duration<double> m_lastInferenceTime;
  std::chrono::duration<double> m_lastPrefillTime;
};
}  // namespace rwkv_app
}  // namespace tools
}  // namespace qnn
