#include <inttypes.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#ifndef _WIN32
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#endif
#include "half.hpp"
#include "DataUtil.hpp"
#include "Logger.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#include "PAL/StringOp.hpp"
#include "QnnTypeMacros.hpp"
#include "librwkv-qualcomm-app.hpp"
#include "Utils.hpp"
#include "QnnWrapperUtils.hpp"
#include "IOTensor.hpp"
#include "DynamicLoadUtil.hpp"
#include "PAL/DynamicLoading.hpp"
#include <stdlib.h>

#include <HTP/QnnHtpPerfInfrastructure.h>
#include <QnnInterface.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpContext.h>
#include <QnnContext.h>

#ifndef _WIN32
#define USE_MMAP 1
#else
#define USE_MMAP 0
#endif

using namespace qnn;
using namespace qnn::tools;

std::string defaultOutputPath = "./output";

rwkv_app::QnnRwkvApp::QnnRwkvApp(QnnFunctionPointers qnnFunctionPointers,
                                       void* backendLibraryHandle,
                                       void* modelHandle,
                                       std::vector<std::vector<float>> embedding,
                                       std::string cachedBinaryPath,
                                       std::string saveBinaryName)
    : m_qnnFunctionPointers(qnnFunctionPointers),
      m_saveBinaryName(saveBinaryName),
      m_cachedBinaryPath(cachedBinaryPath),
      m_backendLibraryHandle(backendLibraryHandle),
      m_modelHandle(modelHandle),
      m_isBackendInitialized(false),
      m_isContextCreated(false) {
  m_embedding = embedding;
  m_outputPath = defaultOutputPath;

  m_ioTensor = new IOTensor(BufferAlloc::SHARED_BUFFER, &m_qnnFunctionPointers.qnnInterface);
  return;
}

rwkv_app::QnnRwkvApp::~QnnRwkvApp() {
  if (StatusCode::SUCCESS != freeGraphs()) {
    QNN_ERROR("Could not free graphs.");
  }

  if (StatusCode::SUCCESS != freeContext()) {
    QNN_ERROR("Could not free context.");
  }

  if (StatusCode::SUCCESS != destroyPowerConfigId()) {
    QNN_ERROR("Could not destroy power config id.");
  }

  auto devicePropertySupportedStatus = isDevicePropertySupported();
  if (StatusCode::FAILURE != devicePropertySupportedStatus) {
    auto freeDeviceStatus = freeDevice();
    if (StatusCode::FAILURE == freeDeviceStatus) {
      QNN_ERROR("Could not free device.");
    }
  }

  if (m_backendLibraryHandle)
    pal::dynamicloading::dlClose(m_backendLibraryHandle);

  if (m_modelHandle)
    pal::dynamicloading::dlClose(m_modelHandle);

  // Terminate backend
  if (m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) {
    QNN_DEBUG("Freeing backend");
    if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
      QNN_ERROR("Could not free backend");
    }
  }
  m_isBackendInitialized = false;
  // Terminate logging in the backend
  if (nullptr != m_qnnFunctionPointers.qnnInterface.logFree && nullptr != m_logHandle) {
    if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.logFree(m_logHandle)) {
      QNN_WARN("Unable to terminate logging in the backend.");
    }
  }
  return;
}

std::string rwkv_app::QnnRwkvApp::getBackendBuildId() {
  char* backendBuildId{nullptr};
  if (QNN_SUCCESS !=
      m_qnnFunctionPointers.qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
    QNN_ERROR("Unable to get build Id from the backend.");
  }
  return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::initialize() {
  // initialize logging in the backend
  if (log::isLogInitialized()) {
    auto logCallback = log::getLogCallback();
    auto logLevel    = log::getLogLevel();
    QNN_INFO("Initializing logging in the backend. Callback: [%p], Log Level: [%d]",
             logCallback,
             logLevel);
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
      QNN_WARN("Unable to initialize logging in the backend.");
    }
  } else {
    QNN_WARN("Logging not available in the backend.");
  }
  return StatusCode::SUCCESS;
}

// Initialize a QnnBackend.
rwkv_app::StatusCode rwkv_app::QnnRwkvApp::initializeBackend() {
  auto qnnStatus = m_qnnFunctionPointers.qnnInterface.backendCreate(
      m_logHandle, (const QnnBackend_Config_t**)m_backendConfig, &m_backendHandle);
  if (QNN_BACKEND_NO_ERROR != qnnStatus) {
    QNN_ERROR("Could not initialize backend due to error = %d", qnnStatus);
    return StatusCode::FAILURE;
  }
  QNN_INFO("Initialize Backend Returned Status = %d", qnnStatus);
  m_isBackendInitialized = true;
  return StatusCode::SUCCESS;
}

// Terminate the backend after done.
rwkv_app::StatusCode rwkv_app::QnnRwkvApp::terminateBackend() {
  if ((m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) &&
      QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
    QNN_ERROR("Could not terminate backend");
    return StatusCode::FAILURE;
  }
  m_isBackendInitialized = false;
  return StatusCode::SUCCESS;
}

// Register op packages and interface providers supplied during
// object creation. If there are multiple op packages, register
// them sequentially in the order provided.
rwkv_app::StatusCode rwkv_app::QnnRwkvApp::registerOpPackages() {
  const size_t pathIdx              = 0;
  const size_t interfaceProviderIdx = 1;
  for (auto const& opPackagePath : m_opPackagePaths) {
    std::vector<std::string> opPackage;
    split(opPackage, opPackagePath, ':');
    QNN_DEBUG("opPackagePath: %s", opPackagePath.c_str());
    const char* target     = nullptr;
    const size_t targetIdx = 2;
    if (opPackage.size() != 2 && opPackage.size() != 3) {
      QNN_ERROR("Malformed opPackageString provided: %s", opPackagePath.c_str());
      return StatusCode::FAILURE;
    }
    if (opPackage.size() == 3) {
      target = (char*)opPackage[targetIdx].c_str();
    }
    if (nullptr == m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
      QNN_ERROR("backendRegisterOpPackageFnHandle is nullptr.");
      return StatusCode::FAILURE;
    }
    if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
                                    m_backendHandle,
                                    (char*)opPackage[pathIdx].c_str(),
                                    (char*)opPackage[interfaceProviderIdx].c_str(),
                                    target)) {
      QNN_ERROR("Could not register Op Package: %s and interface provider: %s",
                opPackage[pathIdx].c_str(),
                opPackage[interfaceProviderIdx].c_str());
      return StatusCode::FAILURE;
    }
    QNN_INFO("Registered Op Package: %s and interface provider: %s",
             opPackage[pathIdx].c_str(),
             opPackage[interfaceProviderIdx].c_str());
  }
  return StatusCode::SUCCESS;
}

// Create a Context in a backend.
rwkv_app::StatusCode rwkv_app::QnnRwkvApp::createContext() {
  if (QNN_CONTEXT_NO_ERROR != m_qnnFunctionPointers.qnnInterface.contextCreate(
                                  m_backendHandle,
                                  m_deviceHandle,
                                  (const QnnContext_Config_t**)m_contextConfig,
                                  &m_context[0])) {
    QNN_ERROR("Could not create context");
    return StatusCode::FAILURE;
  }
  m_isContextCreated = true;
  return StatusCode::SUCCESS;
}

// Free context after done.
rwkv_app::StatusCode rwkv_app::QnnRwkvApp::freeContext() {
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextFree(m_context[0], nullptr)) {
    QNN_ERROR("Could not free context");
    return StatusCode::FAILURE;
  }
  m_isContextCreated = false;
  return StatusCode::SUCCESS;
}

// Calls composeGraph function in QNN's model.so.
// composeGraphs is supposed to populate graph related
// information in m_decodeGraphsInfo and m_decodeGraphsCount.
// m_debug is the option supplied to composeGraphs to
// say that all intermediate tensors including output tensors
// are expected to be read by the app.
rwkv_app::StatusCode rwkv_app::QnnRwkvApp::composeGraphs() {
  if (m_graphConfigsInfo == nullptr) {
    m_graphConfigsInfoCount = 3;
    m_graphConfigsInfo = new GraphConfigInfo_t*[m_graphConfigsInfoCount];
    m_graphConfigsInfo[0] = new GraphConfigInfo_t();
    m_graphConfigsInfo[0]->graphName = (char*)"model";
    m_graphConfigsInfo[0]->graphConfigs = (const QnnGraph_Config_t**)new QnnGraph_Config_t*[2];
    m_graphConfigsInfo[1] = new GraphConfigInfo_t();
    m_graphConfigsInfo[1]->graphName = (char*)"RWKV_6_ABC_85M_v1_20240217_ctx1024";
    m_graphConfigsInfo[1]->graphConfigs = (const QnnGraph_Config_t**)new QnnGraph_Config_t*[2];
    m_graphConfigsInfo[2] = new GraphConfigInfo_t();
    m_graphConfigsInfo[2]->graphName = (char*)"sudoku_rwkv_20241120";
    m_graphConfigsInfo[2]->graphConfigs = (const QnnGraph_Config_t**)new QnnGraph_Config_t*[2];

    static QnnHtpGraph_CustomConfig_t customConfig;
    customConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    customConfig.precision = QNN_PRECISION_FLOAT16;
    static QnnGraph_Config_t graphConfig;
    graphConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graphConfig.customConfig = &customConfig;
    m_graphConfigsInfo[0]->graphConfigs[0] = &graphConfig;
    m_graphConfigsInfo[0]->graphConfigs[1] = nullptr;
    m_graphConfigsInfo[1]->graphConfigs[0] = &graphConfig;
    m_graphConfigsInfo[1]->graphConfigs[1] = nullptr;
    m_graphConfigsInfo[2]->graphConfigs[0] = &graphConfig;
    m_graphConfigsInfo[2]->graphConfigs[1] = nullptr;
  }

  auto returnStatus = StatusCode::SUCCESS;
  if (ModelError_t::MODEL_NO_ERROR !=
      m_qnnFunctionPointers.composeGraphsFnHandle(
          m_backendHandle,
          m_qnnFunctionPointers.qnnInterface,
          m_context[0],
          (const GraphConfigInfo_t**)m_graphConfigsInfo,
          m_graphConfigsInfoCount,
          &m_decodeGraphsInfo,
          &m_decodeGraphsCount,
          false,
          log::getLogCallback(),
          log::getLogLevel())) {
    QNN_ERROR("Failed in composeGraphs()");
    returnStatus = StatusCode::FAILURE;
  }
  return returnStatus;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::finalizeGraphs() {
  // no weight sharing for online preparing graphs
  for (size_t graphIdx = 0; graphIdx < m_decodeGraphsCount; graphIdx++) {
    if (QNN_GRAPH_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.graphFinalize(
            (*m_decodeGraphsInfo)[graphIdx].graph, nullptr, nullptr)) {
      return StatusCode::FAILURE;
    }
  }
  auto returnStatus = StatusCode::SUCCESS;
  if (!m_saveBinaryName.empty()) {
    QNN_INFO("Before saveBinary(): saving context and metadata.");
    returnStatus = saveBinary();
  } else {
    QNN_DEBUG("m_saveBinaryName is empty()");
  }
  return returnStatus;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::createFromBinary(uint8_t *in_buffer, uint64_t bufferSize) {
  if (m_cachedBinaryPath.empty() && (nullptr == in_buffer || 0 == bufferSize)) {
    QNN_ERROR("No name provided to read binary file from.");
    return StatusCode::FAILURE;
  }
  if (nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
    QNN_ERROR("QNN System function pointers are not populated.");
    return StatusCode::FAILURE;
  }

  std::vector<std::shared_ptr<uint8_t>> buffer;
  std::vector<uint64_t> bufferSizes;
  auto pos = m_cachedBinaryPath.find("_chunk");
  int n_chunks = 1;
  if (pos != std::string::npos) {
    n_chunks = std::stoi(m_cachedBinaryPath.substr(m_cachedBinaryPath.find("of") + 2));
    QNN_INFO("Number of chunks: %d", n_chunks);
  }
  buffer.resize(n_chunks);
  bufferSizes.resize(n_chunks);

  if (in_buffer && bufferSize) {
    buffer[0] = std::shared_ptr<uint8_t>(in_buffer);
    bufferSizes[0] = bufferSize;
  } else {
    // read serialized binary into a byte buffer
    tools::datautil::StatusCode status{tools::datautil::StatusCode::SUCCESS};
    for (int i = 0; i < n_chunks; i++) {
      if (n_chunks > 1) {
        m_cachedBinaryPath = m_cachedBinaryPath.substr(0, pos) + "_chunk" + std::to_string(i+1) + "of" + std::to_string(n_chunks) + ".bin";
        std::cout << "Reading chunk: " << m_cachedBinaryPath << std::endl;
      }
      std::tie(status, bufferSizes[i]) = tools::datautil::getFileSize(m_cachedBinaryPath);
      if (0 == bufferSizes[i]) {
        QNN_ERROR("Received path to an empty file. Nothing to deserialize.");
        return StatusCode::FAILURE;
      }
      std::cout << "Buffer size: " << bufferSizes[i] << std::endl;

#if USE_MMAP
      int fd = open(m_cachedBinaryPath.c_str(), O_RDONLY);
      if (fd < 0) {
        QNN_ERROR("Failed to open file %s", m_cachedBinaryPath.c_str());
        return StatusCode::FAILURE;
      }

      buffer[i] = std::shared_ptr<uint8_t>(
          (uint8_t*)mmap(NULL, bufferSizes[i], PROT_READ, MAP_SHARED, fd, 0), [bufferSizes, i](uint8_t* p) {
              if (p) {
                munmap(p, bufferSizes[i]);
              }
            }
          );

      if (buffer[i].get() == MAP_FAILED) {
        QNN_ERROR("Failed to mmap file %s", m_cachedBinaryPath.c_str());
        close(fd);
        return StatusCode::FAILURE;
      }
#else
      buffer[i] = std::shared_ptr<uint8_t>(new uint8_t[bufferSizes[i]], std::default_delete<uint8_t[]>());
      if (!buffer[i]) {
        QNN_ERROR("Failed to allocate memory.");
        return StatusCode::FAILURE;
      }

      status = tools::datautil::readBinaryFromFile(
          m_cachedBinaryPath, reinterpret_cast<uint8_t*>(buffer[i].get()), bufferSizes[i]);
      if (status != tools::datautil::StatusCode::SUCCESS) {
        QNN_ERROR("Failed to read binary data.");
        return StatusCode::FAILURE;
      }
#endif
    }
  }

  // inspect binary info
  auto returnStatus = StatusCode::SUCCESS;
  std::vector<GraphInfo_t **> graphInfos(n_chunks);
  std::vector<uint32_t> graphCounts(n_chunks);
  for (int i = 0; i < n_chunks; i++)
  {
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS != m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
      QNN_ERROR("Could not create system handle.");
      returnStatus = StatusCode::FAILURE;
    }

    const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};
    if (StatusCode::SUCCESS == returnStatus &&
        QNN_SUCCESS != m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                          sysCtxHandle,
                          static_cast<void*>(buffer[i].get()),
                          bufferSizes[i],
                          &binaryInfo,
                          &binaryInfoSize)) {
      QNN_ERROR("Failed to get context binary info");
      returnStatus = StatusCode::FAILURE;
    }

    // fill GraphInfo_t based on binary info
    if (StatusCode::SUCCESS == returnStatus &&
        !copyMetadataToGraphsInfo(binaryInfo, graphInfos[i], graphCounts[i])) {
      QNN_ERROR("Failed to copy metadata.");
      returnStatus = StatusCode::FAILURE;
    }
    m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (StatusCode::SUCCESS == returnStatus &&
        nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
      QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr.");
      returnStatus = StatusCode::FAILURE;
    }

    // QnnHtpContext_CustomConfig_t customConfig;
    // customConfig.option = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
    // customConfig.ioMemEstimation = true;
    // QnnContext_Config_t* cfgs[] = {(QnnContext_Config_t*)&customConfig, NULL};

    if (StatusCode::SUCCESS == returnStatus &&
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
            m_backendHandle,
            m_deviceHandle,
            // (const QnnContext_Config_t**)cfgs,
            (const QnnContext_Config_t**)m_contextConfig,
            static_cast<void*>(buffer[i].get()),
            bufferSizes[i],
            &m_context[i],
            nullptr)) {
      QNN_ERROR("Could not create context from binary.");
      returnStatus = StatusCode::FAILURE;
    }
    m_isContextCreated = true;
    if (StatusCode::SUCCESS == returnStatus) {
      for (size_t graphIdx = 0; graphIdx < graphCounts[i]; graphIdx++) {
        if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
          QNN_ERROR("graphRetrieveFnHandle is nullptr.");
          returnStatus = StatusCode::FAILURE;
          break;
        }
        if (QNN_SUCCESS !=
            m_qnnFunctionPointers.qnnInterface.graphRetrieve(
                m_context[i], (*graphInfos[i])[graphIdx].graphName, &((*graphInfos[i])[graphIdx].graph))) {
          QNN_ERROR("Unable to retrieve graph handle for graph Idx: %d", graphIdx);
          returnStatus = StatusCode::FAILURE;
        }
      }
    }
    if (StatusCode::SUCCESS != returnStatus) {
      QNN_DEBUG("Cleaning up graph Info structures.");
      freeGraphsInfo(&graphInfos[i], graphCounts[i]);
    }
  }

  buffer.clear();

  m_decodeGraphsCount = 0;
  m_prefillGraphsCount = 0;
  for (int i = 0; i < n_chunks; i++) {
    for (int j = 0; j < graphCounts[i]; j++) {
      auto graphName = std::string((*graphInfos[i])[j].graphName);
      if (graphName.find("prefill") != std::string::npos) {
        m_prefillGraphsCount++;
      } else {
        m_decodeGraphsCount++;
      }
    }
  }
  QNN_INFO("Decode graphs count: %d, Prefill graphs count: %d", m_decodeGraphsCount, m_prefillGraphsCount);

  m_decodeGraphsInfo = (GraphInfo_t **)calloc(m_decodeGraphsCount, sizeof(GraphInfo_t *));
  m_prefillGraphsInfo = (GraphInfo_t **)calloc(m_prefillGraphsCount, sizeof(GraphInfo_t *));
  GraphInfo_t *decodeGraphInfoArr =
      (GraphInfo_t *)calloc(m_decodeGraphsCount, sizeof(GraphInfo_t));
  GraphInfo_t *prefillGraphInfoArr =
      (GraphInfo_t *)calloc(m_prefillGraphsCount, sizeof(GraphInfo_t));
  if (nullptr == m_decodeGraphsInfo || nullptr == m_prefillGraphsInfo || nullptr == decodeGraphInfoArr || nullptr == prefillGraphInfoArr) {
    QNN_ERROR("Failure to allocate memory for *graphInfo");
    if (nullptr != m_decodeGraphsInfo) {
      free(m_decodeGraphsInfo);
      m_decodeGraphsInfo = nullptr;
    }
    if (nullptr != m_prefillGraphsInfo) {
      free(m_prefillGraphsInfo);
      m_prefillGraphsInfo = nullptr;
    }
    if (nullptr != decodeGraphInfoArr) {
      free(decodeGraphInfoArr);
      decodeGraphInfoArr = nullptr;
    }
    if (nullptr != prefillGraphInfoArr) {
      free(prefillGraphInfoArr);
      prefillGraphInfoArr = nullptr;
    }
  }

  if (StatusCode::SUCCESS == returnStatus) {
    int prefill_gidx = 0, decode_gidx = 0;
    for (int i = 0; i < n_chunks; i++) {
      for (int j = 0; j < graphCounts[i]; j++) {
        auto graphName = std::string((*graphInfos[i])[j].graphName);
        if (graphName.find("prefill") != std::string::npos) {
          m_prefillGraphsInfo[prefill_gidx] = prefillGraphInfoArr + prefill_gidx;
          m_prefillGraphsInfo[prefill_gidx]->graph = (*graphInfos[i])[j].graph;
          m_prefillGraphsInfo[prefill_gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
          m_prefillGraphsInfo[prefill_gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
          m_prefillGraphsInfo[prefill_gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
          m_prefillGraphsInfo[prefill_gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
          m_prefillGraphsInfo[prefill_gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
          prefill_gidx++;
        } else {
          m_decodeGraphsInfo[decode_gidx] = decodeGraphInfoArr + decode_gidx;
          m_decodeGraphsInfo[decode_gidx]->graph = (*graphInfos[i])[j].graph;
          m_decodeGraphsInfo[decode_gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
          m_decodeGraphsInfo[decode_gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
          m_decodeGraphsInfo[decode_gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
          m_decodeGraphsInfo[decode_gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
          m_decodeGraphsInfo[decode_gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
          decode_gidx++;
        }
      }
    }
  }
  return returnStatus;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::saveBinary() {
  if (m_saveBinaryName.empty()) {
    QNN_ERROR("No name provided to save binary file.");
    return StatusCode::FAILURE;
  }
  if (nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinarySize ||
      nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinary) {
    QNN_ERROR("contextGetBinarySizeFnHandle or contextGetBinaryFnHandle is nullptr.");
    return StatusCode::FAILURE;
  }
  uint64_t requiredBufferSize{0};
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextGetBinarySize(m_context[0], &requiredBufferSize)) {
    QNN_ERROR("Could not get the required binary size.");
    return StatusCode::FAILURE;
  }
  std::unique_ptr<uint8_t[]> saveBuffer(new uint8_t[requiredBufferSize]);
  if (nullptr == saveBuffer) {
    QNN_ERROR("Could not allocate buffer to save binary.");
    return StatusCode::FAILURE;
  }
  uint64_t writtenBufferSize{0};
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextGetBinary(m_context[0],
                                                          reinterpret_cast<void*>(saveBuffer.get()),
                                                          requiredBufferSize,
                                                          &writtenBufferSize)) {
    QNN_ERROR("Could not get binary.");
    return StatusCode::FAILURE;
  }
  if (requiredBufferSize < writtenBufferSize) {
    QNN_ERROR(
        "Illegal written buffer size [%d] bytes. Cannot exceed allocated memory of [%d] bytes",
        writtenBufferSize,
        requiredBufferSize);
    return StatusCode::FAILURE;
  }
  auto dataUtilStatus = tools::datautil::writeBinaryToFile(
      m_outputPath, m_saveBinaryName + ".bin", (uint8_t*)saveBuffer.get(), writtenBufferSize);
  if (tools::datautil::StatusCode::SUCCESS != dataUtilStatus) {
    QNN_ERROR("Error while writing binary to file.");
    return StatusCode::FAILURE;
  }
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::verifyFailReturnStatus(Qnn_ErrorHandle_t errCode) {
  auto returnStatus = rwkv_app::StatusCode::FAILURE;
  switch (errCode) {
    case QNN_COMMON_ERROR_SYSTEM_COMMUNICATION:
      returnStatus = rwkv_app::StatusCode::FAILURE_SYSTEM_COMMUNICATION_ERROR;
      break;
    case QNN_COMMON_ERROR_SYSTEM:
      returnStatus = rwkv_app::StatusCode::FAILURE_SYSTEM_ERROR;
      break;
    case QNN_COMMON_ERROR_NOT_SUPPORTED:
      returnStatus = rwkv_app::StatusCode::QNN_FEATURE_UNSUPPORTED;
      break;
    default:
      break;
  }
  return returnStatus;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::isDevicePropertySupported() {
  if (nullptr != m_qnnFunctionPointers.qnnInterface.propertyHasCapability) {
    auto qnnStatus =
        m_qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
      QNN_WARN("Device property is not supported");
    }
    if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
      QNN_ERROR("Device property is not known to backend");
      return StatusCode::FAILURE;
    }
  }
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::createDevice() {
  if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceCreate) {
    auto qnnStatus =
        m_qnnFunctionPointers.qnnInterface.deviceCreate(m_logHandle, nullptr, &m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
      QNN_ERROR("Failed to create device");
      return verifyFailReturnStatus(qnnStatus);
    }
  }
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::freeDevice() {
  if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceFree) {
    auto qnnStatus = m_qnnFunctionPointers.qnnInterface.deviceFree(m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
      QNN_ERROR("Failed to free device");
      return verifyFailReturnStatus(qnnStatus);
    }
  }
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::createPowerConfigId() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = m_qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
      QNN_ERROR("device error");
      return StatusCode::FAILURE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;
    Qnn_ErrorHandle_t perfInfraErr = perfInfra.createPowerConfigId(deviceId, coreId, &powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("createPowerConfigId failed");
      return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::setPowerConfig() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = m_qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        QNN_ERROR("device error");
        return StatusCode::FAILURE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
    memset(&powerConfig, 0, sizeof(powerConfig));
    powerConfig.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    powerConfig.dcvsV3Config.dcvsEnable    = 0; //True to enable Dcvs, False to disbale
    powerConfig.dcvsV3Config.setDcvsEnable = 1;
    powerConfig.dcvsV3Config.contextId     = powerConfigId;  //use the power config id created

    // refer QnnHtpPerfInfrastructure.h
    powerConfig.dcvsV3Config.powerMode       = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    powerConfig.dcvsV3Config.setSleepLatency = 1; //True to consider Latency parameter otherwise False
    powerConfig.dcvsV3Config.setBusParams    = 1; //True to consider Bus parameter otherwise False
    powerConfig.dcvsV3Config.setCoreParams   = 1; //True to consider Core parameter otherwise False
    powerConfig.dcvsV3Config.sleepDisable    = 1; //True to disable sleep, False to re-enable sleep
    powerConfig.dcvsV3Config.setSleepDisable = 1; //True to consider sleep disable/enable parameter otherwise False

    //Set Sleep latency parameter
    powerConfig.dcvsV3Config.sleepLatency    =  40; // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk

    //set Bus Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    powerConfig.dcvsV3Config.busVoltageCornerMin     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerTarget  = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerMax     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    //set Core Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    powerConfig.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfigHMX;
    memset(&powerConfigHMX, 0, sizeof(powerConfigHMX));
    powerConfigHMX.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
    powerConfigHMX.hmxV2Config.hmxPickDefault = 0;
    powerConfigHMX.hmxV2Config.hmxPerfMode    = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

    powerConfigHMX.hmxV2Config.hmxVoltageCornerMin    = DCVS_EXP_VCORNER_TUR;
    powerConfigHMX.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_TUR;
    powerConfigHMX.hmxV2Config.hmxVoltageCornerMax    = DCVS_EXP_VCORNER_TUR;

    // Set power config with different performance parameters
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&powerConfig, &powerConfigHMX, NULL};

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs);
    if (perfInfraErr != QNN_SUCCESS) {
        QNN_ERROR("setPowerConfig failed");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::destroyPowerConfigId() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = m_qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        QNN_ERROR("device error");
        return StatusCode::FAILURE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.destroyPowerConfigId(powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
        QNN_ERROR("destroyPowerConfigId failed");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

size_t rwkv_app::QnnRwkvApp::getQnnDatatypeSize(Qnn_DataType_t dataType) {
    switch (dataType) {
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_INT_16:
            return sizeof(uint16_t);
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
            return sizeof(uint32_t);
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_BOOL_8:
            return sizeof(uint8_t);
        case QNN_DATATYPE_FLOAT_64:
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
            return sizeof(uint64_t);
        default:
            QNN_ERROR("Unsupported data type");
            return 0;
    }
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::setRpcLatencyAndPolling() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = m_qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
      QNN_ERROR("device error");
      return StatusCode::FAILURE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    // set RPC Control Latency
    QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;            // refer QnnHtpPerfInfrastructure.h
    memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
    rpcControlLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControlLatency.rpcControlLatencyConfig = 100;         // use rpc control latency recommended 100 us, refer hexagon sdk
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs1[] = {&rpcControlLatency, NULL};

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs1);  // set RPC latency config on power config id created
    if (perfInfraErr != QNN_SUCCESS) {
        QNN_ERROR("setPowerConfig failed");
        return StatusCode::FAILURE;
    }

    // set RPC Polling
    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;   // refer QnnHtpPerfInfrastructure.h
    memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
    rpcPollingTime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingTime.rpcPollingTimeConfig = 9999;     // use rpc polling time recommended 0-10000 us
    const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs2[] = {&rpcPollingTime, NULL};

    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs2); // set RPC polling config on power config id created
    if (perfInfraErr != QNN_SUCCESS) {
        QNN_ERROR("setPowerConfig failed");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

void rwkv_app::QnnRwkvApp::fillQuantizedTensor(float value, Qnn_Tensor_t *tensor) {
  std::vector<size_t> dims;
  for (int j = 0; j < QNN_TENSOR_GET_RANK(*tensor); j++) {
    dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(*tensor) + j));
  }
  void *buffer = m_ioTensor->getBuffer(tensor);
  float fpzero = 0.0;
  auto dtype = QNN_TENSOR_GET_DATA_TYPE(*tensor);
  if (dtype == QNN_DATATYPE_UFIXED_POINT_8) {
    uint8_t qtzero = 0;
    datautil::floatToTfN<uint8_t>(&qtzero, &fpzero,
        QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.offset,
        QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.scale,
        1);
    for (int j = 0; j < datautil::calculateElementCount(dims); j++) {
      ((uint8_t*)buffer)[j] = qtzero;
    }
  } else if (dtype == QNN_DATATYPE_UFIXED_POINT_16) {
    uint16_t qtzero = 0;
    datautil::floatToTfN<uint16_t>(&qtzero, &fpzero,
        QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.offset,
        QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.scale,
        1);
    for (int j = 0; j < datautil::calculateElementCount(dims); j++) {
      ((uint16_t*)buffer)[j] = qtzero;
    }
  }
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::initializeTensors() {
  if (!m_tensorsInitialized) {
    m_ioTensor->initialize(m_context[0]);
    m_decodeGraphsTensorNameToTensorPointer.resize(m_decodeGraphsCount);
    m_decodeGraphsTensorNameToSize.resize(m_decodeGraphsCount);
    if (m_prefillGraphsCount > 0) {
      m_prefillGraphsTensorNameToTensorPointer.resize(m_prefillGraphsCount);
      m_prefillGraphsTensorNameToSize.resize(m_prefillGraphsCount);
    }

    for (int graph_id = 0; graph_id < m_decodeGraphsCount; graph_id++) {
      auto graphInfo     = (*m_decodeGraphsInfo)[graph_id];
      QNN_INFO("Graph %d : %s", graph_id, graphInfo.graphName);
      for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
        size_t tensorDataSize = 1;
        for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.outputTensors[i]); j++) {
          tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.outputTensors[i]) + j);
        }
        auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
        size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]));
        if (typeSize == 0) {
            return StatusCode::FAILURE;
        }
        tensorDataSize *= typeSize;
        m_decodeGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
        QNN_INFO("Output Tensor %d : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]), tensorDataSize);
      }

      if (!m_ioTensor->setupOutputTensors(&m_outputTensors[graph_id], m_decodeGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                          m_decodeGraphsTensorNameToSize[graph_id], m_context[graph_id], false)) {
        QNN_ERROR("Error in setting up Output Tensors");
        return StatusCode::FAILURE;
      }

      std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap;
      for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
        size_t tensorDataSize = 1;
        for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.inputTensors[i]); j++) {
          tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.inputTensors[i]) + j);
        }
        auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.inputTensors[i]));
        size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]));
        if (typeSize == 0) {
            return StatusCode::FAILURE;
        }
        tensorDataSize *= typeSize;
        m_decodeGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
        QNN_INFO("Input Tensor %d : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]), tensorDataSize);
        if (tensorName.find("state") != std::string::npos) {
          sharedTensorMap[tensorName] = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id][tensorName.substr(0, tensorName.find("_in")) + "_out"];
        }
        if (graph_id > 0) {
          if (tensorName.find("v_first_in") != std::string::npos) {
            sharedTensorMap[tensorName] = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[0]["v_first_out_chunk1"];
          } else if (tensorName == "in_chunk" + std::to_string(graph_id + 1)) {
            sharedTensorMap[tensorName] = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id - 1]["out_chunk" + std::to_string(graph_id)];
          }
        }
      }

      if (!m_ioTensor->setupInputWithSharedTensors(&m_inputTensors[graph_id], m_decodeGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                        m_decodeGraphsTensorNameToSize[graph_id], m_context[graph_id], sharedTensorMap)) {
        QNN_ERROR("Error in setting up Input Tensors");
        return StatusCode::FAILURE;
      }

      for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
        // fill state tensors with zeros
        std::vector<size_t> dims;
        for (int j = 0; j < QNN_TENSOR_GET_RANK(m_outputTensors[graph_id][i]); j++) {
          dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(m_outputTensors[graph_id][i]) + j));
        }
        void *buffer = m_ioTensor->getBuffer(&m_outputTensors[graph_id][i]);
        if (QNN_TENSOR_GET_DATA_TYPE(m_outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_16)
          memset(buffer, 0, datautil::calculateElementCount(dims) * sizeof(uint16_t));
        else if (QNN_TENSOR_GET_DATA_TYPE(m_outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_32)
          memset(buffer, 0, datautil::calculateElementCount(dims) * sizeof(float));
        else {
          fillQuantizedTensor(0.0, &m_outputTensors[graph_id][i]);
        }
      }
    }

    if (m_prefillGraphsCount > 0) {
      std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMapPrefill;
      for (int graph_id = 0; graph_id < m_prefillGraphsCount; graph_id++) {
        auto graphInfo     = (*m_prefillGraphsInfo)[graph_id];
        QNN_INFO("Graph %d : %s", graph_id, graphInfo.graphName);
        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
          size_t tensorDataSize = 1;
          for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.outputTensors[i]); j++) {
            tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.outputTensors[i]) + j);
          }
          auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
          size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]));
          if (typeSize == 0) {
              return StatusCode::FAILURE;
          }
          tensorDataSize *= typeSize;
          m_prefillGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
          QNN_INFO("Output Tensor %d : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]), tensorDataSize);

          if (tensorName.find("state") != std::string::npos) {
            sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id][tensorName];
          } else if (tensorName == "out_prefill") {
            sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id]["out"];
            m_logitsOutputTensor = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id]["out"];
          } else if (graph_id == m_prefillGraphsCount - 1 && tensorName.find("out_prefill_chunk") != std::string::npos) {
            sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id]["out_chunk" + std::to_string(graph_id+1)];
            m_logitsOutputTensor = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id]["out_chunk" + std::to_string(graph_id+1)];
          }
        }

        for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
          size_t tensorDataSize = 1;
          for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.inputTensors[i]); j++) {
            tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.inputTensors[i]) + j);
          }
          auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.inputTensors[i]));
          size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]));
          if (typeSize == 0) {
              return StatusCode::FAILURE;
          }
          tensorDataSize *= typeSize;
          m_prefillGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
          QNN_INFO("Input Tensor %d : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]), tensorDataSize);

          if (tensorName.find("state") != std::string::npos) {
            sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id][tensorName];
          }

          if (graph_id > 0) {
            if (tensorName.find("v_first_in") != std::string::npos) {
              sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)m_prefillGraphsTensorNameToTensorPointer[0]["v_first_out_prefill_chunk1"];
            } else if (tensorName == "in_prefill_chunk" + std::to_string(graph_id + 1)) {
              sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)m_prefillGraphsTensorNameToTensorPointer[graph_id - 1]["out_prefill_chunk" + std::to_string(graph_id)];
            }
          }
        }

        if (!m_ioTensor->setupOutputWithSharedTensors(&m_prefillOutputTensors[graph_id], m_prefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                          m_prefillGraphsTensorNameToSize[graph_id], m_context[graph_id], sharedTensorMapPrefill)) {
          QNN_ERROR("Error in setting up Output Tensors");
          return StatusCode::FAILURE;
        }

        if (!m_ioTensor->setupInputWithSharedTensors(&m_prefillInputTensors[graph_id], m_prefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                          m_prefillGraphsTensorNameToSize[graph_id], m_context[graph_id], sharedTensorMapPrefill)) {
          QNN_ERROR("Error in setting up Input Tensors");
          return StatusCode::FAILURE;
        }

      }

      // auto tensor = (Qnn_Tensor_t*)m_prefillGraphsTensorNameToTensorPointer[0]["in_prefill"];
      Qnn_Tensor_t *tensor = nullptr;
      if (m_prefillGraphsTensorNameToTensorPointer[0].find("in_prefill") != m_prefillGraphsTensorNameToTensorPointer[0].end()) {
        tensor = (Qnn_Tensor_t*)m_prefillGraphsTensorNameToTensorPointer[0]["in_prefill"];
      } else if (m_prefillGraphsTensorNameToTensorPointer[0].find("in_prefill_chunk1") != m_prefillGraphsTensorNameToTensorPointer[0].end()) {
        tensor = (Qnn_Tensor_t*)m_prefillGraphsTensorNameToTensorPointer[0]["in_prefill_chunk1"];
      }
      m_prefillSequenceLength = 1;
      for (int i = 0; i < QNN_TENSOR_GET_RANK(*tensor); i++) {
        m_prefillSequenceLength *= *(QNN_TENSOR_GET_DIMENSIONS(*tensor) + i);
      }
      QNN_INFO("Prefill sequence length: %d", m_prefillSequenceLength);
    }

    m_tensorsInitialized = true;
  }
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::execute(int token) {
  auto returnStatus = StatusCode::SUCCESS;

  if (!m_tensorsInitialized)
    return StatusCode::FAILURE;

  if (m_embedding.empty()) {
    int *token_input = (int*)m_ioTensor->getBuffer(&m_inputTensors[0][0]);
    *token_input = token;
  } else {
    void *buffer = m_ioTensor->getBuffer(&m_inputTensors[0][0]);
    if (QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0]) == QNN_DATATYPE_FLOAT_16) {
      half_float::half *ptr = (half_float::half*)buffer;
      for (int i = 0; i < m_embedding.size(); i++) {
        ptr[i] = half_float::half(m_embedding[token][i]);
      }
    } else if (QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0]) == QNN_DATATYPE_FLOAT_32) {
      float *ptr = (float*)buffer;
      memcpy(ptr, m_embedding[token].data(), m_embedding[token].size() * sizeof(float));
    } else if (QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0]) == QNN_DATATYPE_UFIXED_POINT_16) {
      std::vector<size_t> dims;
      for (int j = 0; j < QNN_TENSOR_GET_RANK(m_inputTensors[0][0]); j++) {
        dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(m_inputTensors[0][0]) + j));
      }

      datautil::floatToTfN<uint16_t>(static_cast<uint16_t*>(buffer),
                                      m_embedding[token].data(),
                                      QNN_TENSOR_GET_QUANT_PARAMS(m_inputTensors[0][0]).scaleOffsetEncoding.offset,
                                      QNN_TENSOR_GET_QUANT_PARAMS(m_inputTensors[0][0]).scaleOffsetEncoding.scale,
                                      datautil::calculateElementCount(dims));
    }
  }

  for (int graph_id = 0; graph_id < m_decodeGraphsCount; graph_id++) {
    auto graphInfo     = (*m_decodeGraphsInfo)[graph_id];
    std::chrono::high_resolution_clock::time_point infer_start = std::chrono::high_resolution_clock::now();
    auto executeStatus =
        m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                        m_inputTensors[graph_id],
                                                        graphInfo.numInputTensors,
                                                        m_outputTensors[graph_id],
                                                        graphInfo.numOutputTensors,
                                                        nullptr,
                                                        nullptr);
    std::chrono::high_resolution_clock::time_point infer_end = std::chrono::high_resolution_clock::now();
    if (!graph_id)
      m_lastInferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start);
    else
      m_lastInferenceTime += std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
    }
  }

  return returnStatus;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::executeSequence(std::vector<int> &tokens) {
  auto returnStatus = StatusCode::SUCCESS;

  if (!m_embedding.empty()) {
    QNN_ERROR("Unsupported yet.");
    return StatusCode::FAILURE;
  }

  if (!m_tensorsInitialized)
    return StatusCode::FAILURE;

  std::chrono::high_resolution_clock::time_point infer_start = std::chrono::high_resolution_clock::now();
  if (m_prefillSequenceLength == 0) {
    for (int i = 0; i < tokens.size(); i++) {
      if (execute(tokens[i]) != StatusCode::SUCCESS) {
        QNN_ERROR("Execute failed.");
        return StatusCode::FAILURE;
      }
    }
  } else {
    int *token_input = (int*)m_ioTensor->getBuffer(&m_prefillInputTensors[0][0]);
    int idx;
    for (idx = 0; (idx+m_prefillSequenceLength) < tokens.size(); idx += m_prefillSequenceLength) {
      for (int i = 0; i < m_prefillSequenceLength; i++) {
        token_input[i] = tokens[idx + i];
      }

      for (int graph_id = 0; graph_id < m_prefillGraphsCount; graph_id++) {
        auto graphInfo     = (*m_prefillGraphsInfo)[graph_id];
        auto executeStatus =
            m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                            m_prefillInputTensors[graph_id],
                                                            graphInfo.numInputTensors,
                                                            m_prefillOutputTensors[graph_id],
                                                            graphInfo.numOutputTensors,
                                                            nullptr,
                                                            nullptr);

        if (QNN_GRAPH_NO_ERROR != executeStatus) {
          returnStatus = StatusCode::FAILURE;
        }
      }
    }
    for (; idx < tokens.size(); idx++) {
      if (execute(tokens[idx]) != StatusCode::SUCCESS) {
        QNN_ERROR("Execute failed.");
        return StatusCode::FAILURE;
      }
    }
  }

  std::chrono::high_resolution_clock::time_point infer_end = std::chrono::high_resolution_clock::now();
  m_lastInferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start);

  return returnStatus;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::freeGraphs() {
  for (int i = 0; i < m_decodeGraphsCount; i++) {
    auto graphInfo     = (*m_decodeGraphsInfo)[i];
    m_ioTensor->tearDownTensors(m_inputTensors[i], graphInfo.numInputTensors);
    m_ioTensor->tearDownTensors(m_outputTensors[i], graphInfo.numOutputTensors);
    m_inputTensors[i]  = nullptr;
    m_outputTensors[i] = nullptr;
  }

  freeGraphsInfo(&m_decodeGraphsInfo, m_decodeGraphsCount);
  m_decodeGraphsInfo = nullptr;

  for (int i = 0; i < m_prefillGraphsCount; i++) {
    auto graphInfo     = (*m_prefillGraphsInfo)[i];
    m_ioTensor->tearDownTensors(m_prefillInputTensors[i], graphInfo.numInputTensors);
    m_ioTensor->tearDownTensors(m_prefillOutputTensors[i], graphInfo.numOutputTensors);
    m_prefillInputTensors[i] = nullptr;
    m_prefillOutputTensors[i] = nullptr;
  }
  freeGraphsInfo(&m_prefillGraphsInfo, m_prefillGraphsCount);
  m_prefillGraphsInfo = nullptr;

  return StatusCode::SUCCESS;
}
