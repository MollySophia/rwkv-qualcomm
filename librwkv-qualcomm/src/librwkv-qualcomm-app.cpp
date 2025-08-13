#include <inttypes.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <chrono>

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
#include "rmpack.h"
#include <stdlib.h>

#include <HTP/QnnHtpPerfInfrastructure.h>
#include <QnnInterface.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpContext.h>
#include <QnnContext.h>

#include <iostream>

#ifndef _WIN32
#define USE_MMAP 1
#else
#define USE_MMAP 0
#endif

#define USE_SPILL_FILL 1

using namespace qnn;
using namespace qnn::tools;

std::string defaultOutputPath = "./output";

rwkv_app::QnnRwkvApp::QnnRwkvApp(QnnFunctionPointers qnnFunctionPointers,
                                       void* backendLibraryHandle,
                                       void* modelHandle,
                                       std::string cachedBinaryPath,
                                       std::string saveBinaryName)
    : m_qnnFunctionPointers(qnnFunctionPointers),
      m_saveBinaryName(saveBinaryName),
      m_cachedBinaryPath(cachedBinaryPath),
      m_backendLibraryHandle(backendLibraryHandle),
      m_modelHandle(modelHandle),
      m_isBackendInitialized(false),
      m_isContextCreated(false) {
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
  for (int i = 0; i < n_chunks; i++) {
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextFree(m_context[i], nullptr)) {
      QNN_ERROR("Could not free context");
      return StatusCode::FAILURE;
    }
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

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::parseGraphsInfo(std::vector<GraphInfo_t **> &graphInfos, std::vector<uint32_t> &graphCounts) {
  m_decodeGraphsCount = 0;
  m_prefillGraphsCount = 0;
  for (int i = 0; i < graphInfos.size(); i++) {
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
    return StatusCode::FAILURE;
  }

  int prefill_gidx = 0, decode_gidx = 0;
  for (int i = 0; i < graphInfos.size(); i++) {
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
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::createFromBinary(int spill_fill_buffer_size) {
  if (m_cachedBinaryPath.empty()) {
    QNN_ERROR("No name provided to read binary file from.");
    return StatusCode::FAILURE;
  }
  if (nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
    QNN_ERROR("QNN System function pointers are not populated.");
    return StatusCode::FAILURE;
  }

  bool is_rmpack = m_cachedBinaryPath.find(".rmpack") != std::string::npos;
  if (is_rmpack) {
    try {
      m_rmpack = new RMPack(m_cachedBinaryPath);
    } catch (const std::exception& e) {
      QNN_ERROR("Failed to load rmpack: %s", e.what());
      return StatusCode::FAILURE;
    }
  }

#if USE_SPILL_FILL
  Qnn_ContextHandle_t first_contextHandle{nullptr};
  QnnHtpContext_CustomConfig_t customConfigSF;
  customConfigSF.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
#endif

  std::vector<uint64_t> bufferSizes;
  n_chunks = 1;
  auto returnStatus = StatusCode::SUCCESS;
  size_t pos = 0;
  if (is_rmpack) {
    n_chunks = m_rmpack->getConfig()["n_chunks"];
    spill_fill_buffer_size = m_rmpack->getConfig()["spill_fill_buffer_size"];
    m_hidden_size = m_rmpack->getConfig()["hidden_size"];
    m_vocab_size = m_rmpack->getConfig()["vocab_size"];
    int use_external_lmhead = m_rmpack->getConfig()["use_external_lmhead"];
    if (use_external_lmhead) {
      std::string external_lmhead_filetype = m_rmpack->getConfig()["external_lmhead_filetype"];
      if (external_lmhead_filetype != "raw_fp16") {
        QNN_ERROR("Unsupported external lmhead filetype: %s", external_lmhead_filetype.c_str());
        return StatusCode::FAILURE;
      }

      m_lmhead_weight = std::shared_ptr<uint8_t>(
        (uint8_t*)m_rmpack->readFileToMemory("lmhead"), [this](uint8_t* p) {
          if (p) {
            m_rmpack->freeFileMemory("lmhead");
          }
        }
      );
      for (int i = 0; i < 10; i++) {
        std::cout << ((half_float::half*)m_lmhead_weight.get())[i] << " ";
      }
      std::cout << std::endl;

      m_logitsOutput.resize(m_vocab_size);
    }

    QNN_INFO("Number of chunks: %d", n_chunks);
    bufferSizes.resize(n_chunks);
    for (auto rmpack_file : m_rmpack->getFiles()) {
      if (rmpack_file.filename.find("model_") != std::string::npos) {
        int index = std::stoi(rmpack_file.filename.substr(rmpack_file.filename.find("_") + 1));
        bufferSizes[index] = m_rmpack->getFileSize(rmpack_file.filename);
        QNN_INFO("Reading chunk: %d", index);
        QNN_INFO("Buffer size: %d", bufferSizes[index]);
      }
    }
  } else {
    pos = m_cachedBinaryPath.find("_chunk");
    if (pos != std::string::npos) {
      n_chunks = std::stoi(m_cachedBinaryPath.substr(m_cachedBinaryPath.find("of") + 2));
      QNN_INFO("Number of chunks: %d", n_chunks);
    }
    bufferSizes.resize(n_chunks);
    if (n_chunks == 4) {
      spill_fill_buffer_size = 320000000;
    }
    // read serialized binary into a byte buffer
    tools::datautil::StatusCode status{tools::datautil::StatusCode::SUCCESS};
    for (int i = 0; i < n_chunks; i++) {
      std::string tmp_path = m_cachedBinaryPath;
      if (n_chunks > 1) {
        tmp_path = m_cachedBinaryPath.substr(0, pos) + "_chunk" + std::to_string(i+1) + "of" + std::to_string(n_chunks) + ".bin";
        std::cout << "Reading chunk: " << tmp_path << std::endl;
      }
      std::tie(status, bufferSizes[i]) = tools::datautil::getFileSize(tmp_path);
      if (0 == bufferSizes[i]) {
        QNN_ERROR("Received path to an empty file. Nothing to deserialize.");
        return StatusCode::FAILURE;
      }
      std::cout << "Buffer size: " << bufferSizes[i] << std::endl;
    }
  }

  // inspect binary info
  std::vector<GraphInfo_t **> graphInfos(n_chunks);
  std::vector<uint32_t> graphCounts(n_chunks);
  uint8_t *buffer;
  for (int i = 0; i < n_chunks; i++)
  {
    if (is_rmpack) {
#if USE_MMAP
      buffer = (uint8_t*)m_rmpack->mmapFile("model_" + std::to_string(i));
#else
      buffer = (uint8_t*)m_rmpack->readFileToMemory("model_" + std::to_string(i));
#endif
    } else {
      std::string tmp_path = m_cachedBinaryPath;
      if (n_chunks > 1) {
        tmp_path = m_cachedBinaryPath.substr(0, pos) + "_chunk" + std::to_string(i+1) + "of" + std::to_string(n_chunks) + ".bin";
      }
#if USE_MMAP
      int fd = open(tmp_path.c_str(), O_RDONLY);
      if (fd < 0) {
        QNN_ERROR("Failed to open file %s", tmp_path.c_str());
        return StatusCode::FAILURE;
      }

      buffer = (uint8_t*)mmap(NULL, bufferSizes[i], PROT_READ, MAP_SHARED, fd, 0);
      if (buffer == MAP_FAILED) {
        QNN_ERROR("Failed to mmap file %s", tmp_path.c_str());
        close(fd);
        return StatusCode::FAILURE;
      }
#else
      buffer = (uint8_t*)malloc(bufferSizes[i]);
      if (!buffer) {
        QNN_ERROR("Failed to allocate memory.");
        return StatusCode::FAILURE;
      }

      auto status = tools::datautil::readBinaryFromFile(
          m_cachedBinaryPath, buffer, bufferSizes[i]);
      if (status != tools::datautil::StatusCode::SUCCESS) {
        QNN_ERROR("Failed to read binary data.");
        return StatusCode::FAILURE;
      }
#endif
    }

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
                          static_cast<void*>(buffer),
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

    QnnHtpContext_CustomConfig_t ioMemEstimation;
    ioMemEstimation.option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
    ioMemEstimation.ioMemEstimation = true;

    QnnContext_Config_t** cfgs{nullptr};

    int cfgs_count = 1;
#if USE_SPILL_FILL
    if (spill_fill_buffer_size > 0) {
      cfgs_count++;
    }
#endif
    cfgs                  = (QnnContext_Config_t**)malloc((cfgs_count + 1) * sizeof(QnnContext_Config_t*));
    cfgs[0]               = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
    cfgs[0]->option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    cfgs[0]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&ioMemEstimation);
#if USE_SPILL_FILL
    if (spill_fill_buffer_size > 0) {
      QnnHtpContext_GroupRegistration_t groupInfo{nullptr};
      if (i == 0) {
        groupInfo.firstGroupHandle = 0x0;
      } else {
        groupInfo.firstGroupHandle = first_contextHandle;
      }

      groupInfo.maxSpillFillBuffer     = spill_fill_buffer_size;
      customConfigSF.groupRegistration = groupInfo;
      cfgs[cfgs_count-1]               = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
      cfgs[cfgs_count-1]->option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
      cfgs[cfgs_count-1]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&customConfigSF);
    }
#endif
    cfgs[cfgs_count] = nullptr;

    if (StatusCode::SUCCESS == returnStatus &&
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
            m_backendHandle,
            m_deviceHandle,
            (const QnnContext_Config_t**)cfgs,
            // (const QnnContext_Config_t**)m_contextConfig,
            static_cast<void*>(buffer),
            bufferSizes[i],
            &m_context[i],
            nullptr)) {
      QNN_ERROR("Could not create context from binary.");
      returnStatus = StatusCode::FAILURE;
    }

    for (int j = 0; j < cfgs_count; j++) {
      free(cfgs[j]);
      cfgs[j] = nullptr;
    }
    free(cfgs);
    cfgs = nullptr;

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

#if USE_SPILL_FILL
    if (i == 0) {
      first_contextHandle = m_context[i];
    }
#endif

    if (StatusCode::SUCCESS != returnStatus) {
      QNN_DEBUG("Cleaning up graph Info structures.");
      freeGraphsInfo(&graphInfos[i], graphCounts[i]);
    }

    if (is_rmpack) {
#if USE_MMAP
      m_rmpack->unmapFile("model_" + std::to_string(i));
#else
      m_rmpack->freeFileMemory("model_" + std::to_string(i));
#endif
    } else {
#if USE_MMAP
      munmap(buffer, bufferSizes[i]);
#else
      free(buffer);
#endif
    }
  }

  if (StatusCode::SUCCESS == returnStatus) {
    if (rwkv_app::StatusCode::SUCCESS != parseGraphsInfo(graphInfos, graphCounts)) {
      QNN_ERROR("Failed to parse graphs info.");
      returnStatus = StatusCode::FAILURE;
    }
  }

  return returnStatus;
}

void rwkv_app::QnnRwkvApp::contextNotifyFn(Qnn_ContextHandle_t context,
        Qnn_GraphHandle_t graph,
        const char* graph_name,
        QnnContext_createFromBinaryAsyncNotifyType_t completeType,
        void* notifyParam,
        Qnn_ErrorHandle_t /*status*/) {
  std::pair<rwkv_app::QnnRwkvApp*, uint32_t>* pair = reinterpret_cast<std::pair<rwkv_app::QnnRwkvApp*, uint32_t>*>(notifyParam);
  rwkv_app::QnnRwkvApp* rwkv_app     = pair->first;
  uint32_t contextId                 = pair->second;

  if (completeType == QnnContext_createFromBinaryAsyncNotifyType_t::QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT) {
    rwkv_app->updateContext(context, contextId);
  } else if (completeType == QnnContext_createFromBinaryAsyncNotifyType_t::QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT) {
    rwkv_app->updateQnnApiGraphsandContextsInfo(graph_name, graph, contextId);
  }
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::createFromBinaryListAsync() {
  if (m_cachedBinaryPath.empty()) {
    QNN_ERROR("No name provided to read binary file from.");
    return StatusCode::FAILURE;
  }
  if (nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
    QNN_ERROR("QNN System function pointers are not populated.");
    return StatusCode::FAILURE;
  }

  std::vector<uint64_t> bufferSizes;
  auto pos = m_cachedBinaryPath.find("_chunk");
  n_chunks = 1;
  if (pos != std::string::npos) {
    n_chunks = std::stoi(m_cachedBinaryPath.substr(m_cachedBinaryPath.find("of") + 2));
    QNN_INFO("Number of chunks: %d", n_chunks);
  }
  bufferSizes.resize(n_chunks);

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
  }

  // inspect binary info
  auto returnStatus = StatusCode::SUCCESS;
  m_graphInfos.resize(n_chunks);
  m_graphCounts.resize(n_chunks);
  std::vector<std::shared_ptr<uint8_t>> buffer(n_chunks);
  std::vector<QnnContext_Params_t*> context_params_list(n_chunks + 1, nullptr);

  QnnHtpContext_CustomConfig_t ioMemEstimation;
  ioMemEstimation.option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
  ioMemEstimation.ioMemEstimation = true;

  QnnHtpContext_CustomConfig_t shResConfig;
  shResConfig.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES;
  shResConfig.shareResources = true;

  QnnHtpContext_CustomConfig_t shResOptConfig;
  shResOptConfig.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE;
  shResOptConfig.shareResOptType = SEQUENTIAL_WITHOUT_VA_OPTIMIZATION;

  QnnContext_Config_t** cfgs{nullptr};

  int cfgs_count = 3;
  cfgs                  = (QnnContext_Config_t**)malloc((cfgs_count + 1) * sizeof(QnnContext_Config_t*));
  cfgs[0]               = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
  cfgs[0]->option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  cfgs[0]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&ioMemEstimation);
  cfgs[1]               = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
  cfgs[1]->option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  cfgs[1]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&shResConfig);
  cfgs[2]               = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
  cfgs[2]->option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  cfgs[2]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&shResOptConfig);
  cfgs[cfgs_count] = nullptr;

  for (int i = 0; i < n_chunks; i++)
  {
    if (n_chunks > 1) {
      m_cachedBinaryPath = m_cachedBinaryPath.substr(0, pos) + "_chunk" + std::to_string(i+1) + "of" + std::to_string(n_chunks) + ".bin";
    }
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
      buffer[i] = std::shared_ptr<uint8_t>(
        (uint8_t*)malloc(bufferSizes[i]), [bufferSizes, i](uint8_t* p) {
            if (p) {
                free(p);
            }
        }
      );
      if (!buffer[i]) {
        QNN_ERROR("Failed to allocate memory.");
        return StatusCode::FAILURE;
      }

      auto status = tools::datautil::readBinaryFromFile(
          m_cachedBinaryPath, buffer[i].get(), bufferSizes[i]);
      if (status != tools::datautil::StatusCode::SUCCESS) {
        QNN_ERROR("Failed to read binary data.");
        return StatusCode::FAILURE;
      }
#endif
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
        !copyMetadataToGraphsInfo(binaryInfo, m_graphInfos[i], m_graphCounts[i])) {
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

    std::pair<rwkv_app::QnnRwkvApp*, uint32_t>* notifyParam =
        new std::pair<rwkv_app::QnnRwkvApp*, uint32_t>(this, static_cast<size_t>(i));

    QnnContext_Params_t* contextParam = new QnnContext_Params_t{
        .version = QNN_CONTEXT_PARAMS_VERSION_1,
        .v1      = QnnContext_ParamsV1_t{nullptr,
                                    const_cast<const void*>(static_cast<void*>(buffer[i].get())),
                                    bufferSizes[i],
                                    nullptr,
                                    contextNotifyFn,
                                    static_cast<void*>(notifyParam)}};

    context_params_list[i] = contextParam;
  }

  if (nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinaryListAsync) {
    QNN_ERROR("contextCreateFromBinaryListAsyncFnHandle is nullptr");
    for (int i = 0; i < n_chunks; i++) {
      freeGraphsInfo(&m_graphInfos[i], m_graphCounts[i]);
    }
    return StatusCode::FAILURE;
  }

  auto errCode = m_qnnFunctionPointers.qnnInterface.contextCreateFromBinaryListAsync(
    m_backendHandle,
    m_deviceHandle,
    const_cast<const QnnContext_Params_t**>(context_params_list.data()),
    (const QnnContext_Config_t**)cfgs,
    nullptr);

  for (int j = 0; j < cfgs_count; j++) {
    free(cfgs[j]);
    cfgs[j] = nullptr;
  }
  free(cfgs);
  cfgs = nullptr;

  for (int i = 0; i < n_chunks; i++) {
    buffer[i].reset();
  }

  if (errCode == QNN_SUCCESS) {
    m_isContextCreated = true;
  } else {
    QNN_ERROR("Failed to create context from binary list async");
    returnStatus = StatusCode::FAILURE;
  }

  if (StatusCode::SUCCESS != returnStatus) {
    QNN_DEBUG("Cleaning up graph Info structures.");
    for (int i = 0; i < n_chunks; i++) {
      freeGraphsInfo(&m_graphInfos[i], m_graphCounts[i]);
    }
  }

  if (StatusCode::SUCCESS == returnStatus) {
    if (rwkv_app::StatusCode::SUCCESS != parseGraphsInfo(m_graphInfos, m_graphCounts)) {
      QNN_ERROR("Failed to parse graphs info.");
      returnStatus = StatusCode::FAILURE;
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
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigsHMX[] = {&powerConfig, &powerConfigHMX, NULL};

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigsHMX);
    if (perfInfraErr != QNN_SUCCESS) {
      const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&powerConfig, NULL};
      perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs);
      if (perfInfraErr != QNN_SUCCESS) {
        QNN_ERROR("setPowerConfig failed");
        return StatusCode::FAILURE;
      }
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

      for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
        auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
        if (tensorName == "out") {
          m_logitsOutputTensor = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id]["out"];
        } else if (graph_id == m_decodeGraphsCount - 1 && tensorName.find("out_chunk") != std::string::npos) {
          m_logitsOutputTensor = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id]["out_chunk" + std::to_string(graph_id+1)];
        }
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
          } else if (graph_id == m_prefillGraphsCount - 1 && tensorName.find("out_prefill_chunk") != std::string::npos) {
            sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)m_decodeGraphsTensorNameToTensorPointer[graph_id]["out_chunk" + std::to_string(graph_id+1)];
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
      std::string tensorName;
      if (m_prefillGraphsTensorNameToTensorPointer[0].find("in_prefill") != m_prefillGraphsTensorNameToTensorPointer[0].end()) {
        tensor = (Qnn_Tensor_t*)m_prefillGraphsTensorNameToTensorPointer[0]["in_prefill"];
      } else if (m_prefillGraphsTensorNameToTensorPointer[0].find("in_prefill_chunk1") != m_prefillGraphsTensorNameToTensorPointer[0].end()) {
        tensor = (Qnn_Tensor_t*)m_prefillGraphsTensorNameToTensorPointer[0]["in_prefill_chunk1"];
      } else if (m_prefillGraphsTensorNameToTensorPointer[0].find("in_embedding_prefill") != m_prefillGraphsTensorNameToTensorPointer[0].end()) {
        tensor = (Qnn_Tensor_t*)m_prefillGraphsTensorNameToTensorPointer[0]["in_embedding_prefill"];
        tensorName = "in_embedding_prefill";
      } else if (m_prefillGraphsTensorNameToTensorPointer[0].find("in_embedding_prefill_chunk1") != m_prefillGraphsTensorNameToTensorPointer[0].end()) {
        tensor = (Qnn_Tensor_t*)m_prefillGraphsTensorNameToTensorPointer[0]["in_embedding_prefill_chunk1"];
        tensorName = "in_embedding_prefill_chunk1";
      }
      m_prefillSequenceLength = 1;
      if (tensorName == "in_embedding_prefill" || tensorName == "in_embedding_prefill_chunk1") {
        for (int i = 0; i < QNN_TENSOR_GET_RANK(*tensor) - 1; i++) {
          m_prefillSequenceLength *= *(QNN_TENSOR_GET_DIMENSIONS(*tensor) + i);
        }
      } else {
        for (int i = 0; i < QNN_TENSOR_GET_RANK(*tensor); i++) {
          m_prefillSequenceLength *= *(QNN_TENSOR_GET_DIMENSIONS(*tensor) + i);
        }
      }
      QNN_INFO("Prefill sequence length: %d", m_prefillSequenceLength);
    }

    m_tensorsInitialized = true;
  }

  auto file_exists = [](const std::string& path) {
      std::ifstream file(path);
      return file.good();
  };

  bool is_rmpack = m_rmpack != nullptr;
  if (m_embedding == nullptr && QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0]) != QNN_DATATYPE_INT_32) {
    Qnn_DataType_t emb_file_dtype;
    std::string emb_path = "";
    if (is_rmpack) {
      std::string external_embedding_dtype = m_rmpack->getConfig()["external_embedding_dtype"];
      if (external_embedding_dtype == "fp32") {
        emb_file_dtype = QNN_DATATYPE_FLOAT_32;
      } else if (external_embedding_dtype == "fp16") {
        emb_file_dtype = QNN_DATATYPE_FLOAT_16;
      } else if (external_embedding_dtype == "uint16") {
        emb_file_dtype = QNN_DATATYPE_UFIXED_POINT_16;
      }
      emb_path = "embedding";
    } else {
      QNN_INFO("Checking embedding file: %s", (m_cachedBinaryPath.substr(0, m_cachedBinaryPath.find_last_of(".")) + ".uint16.emb").c_str());
      if (file_exists(m_cachedBinaryPath.substr(0, m_cachedBinaryPath.find_last_of(".")) + ".uint16.emb")) {
          emb_file_dtype = QNN_DATATYPE_UFIXED_POINT_16;
          emb_path = m_cachedBinaryPath.substr(0, m_cachedBinaryPath.find_last_of(".")) + ".uint16.emb";
      }
      else if (file_exists(m_cachedBinaryPath.substr(0, m_cachedBinaryPath.find_last_of(".")) + ".fp32.emb")) {
          emb_file_dtype = QNN_DATATYPE_FLOAT_32;
          emb_path = m_cachedBinaryPath.substr(0, m_cachedBinaryPath.find_last_of(".")) + ".fp32.emb";
      }
      else if (file_exists(m_cachedBinaryPath.substr(0, m_cachedBinaryPath.find_last_of(".")) + ".fp16.emb")) {
          emb_file_dtype = QNN_DATATYPE_FLOAT_16;
          emb_path = m_cachedBinaryPath.substr(0, m_cachedBinaryPath.find_last_of(".")) + ".fp16.emb";
      }

      if (emb_path.empty()) {
          QNN_ERROR("Model needs embedding input, but embedding file is not found");
          return StatusCode::FAILURE;
      }
    }

    std::ifstream emb_file;
    if (!is_rmpack) {
      QNN_INFO("Embedding file path: %s", emb_path.c_str());
      emb_file.open(emb_path, std::ios::in|std::ios::binary);
    }
    if (is_rmpack || emb_file.is_open()) {
      int hidden_size = QNN_TENSOR_GET_DIMENSIONS(m_inputTensors[0][0])[QNN_TENSOR_GET_RANK(m_inputTensors[0][0]) - 1];
      size_t file_size = 0;
      if (is_rmpack) {
        file_size = m_rmpack->getFileSize(emb_path);
      } else {
        emb_file.seekg(0, std::ios::end);
        file_size = emb_file.tellg();
        emb_file.seekg(0, std::ios::beg);
      }
      if (file_size <= 0) {
        QNN_ERROR("Embedding file size is 0");
        return StatusCode::FAILURE;
      }
      if (emb_file_dtype == QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0])) {
        if (!is_rmpack) {
#ifndef _WIN32
          int fd = open(emb_path.c_str(), O_RDONLY);
          if (fd < 0) {
              QNN_ERROR("Failed to open file %s", emb_path.c_str());
              return StatusCode::FAILURE;
          }
          m_embedding = std::shared_ptr<uint8_t>(
              (uint8_t*)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0), [file_size](uint8_t* p) {
                  if (p) {
                      munmap(p, file_size);
                  }
              }
          );
          QNN_INFO("mmap embedding success");
#else
          m_embedding = std::shared_ptr<uint8_t>(
              (uint8_t*)malloc(file_size), [file_size](uint8_t* p) {
                  if (p) {
                      free(p);
                  }
              }
          );
          if (!m_embedding) {
              QNN_ERROR("Failed to allocate memory for embedding");
              return StatusCode::FAILURE;
          }
          emb_file.read(reinterpret_cast<char*>(m_embedding.get()), file_size);
          LOG_ERROR("malloc embedding success");
#endif
        } else {
#ifndef _WIN32
          m_embedding = std::shared_ptr<uint8_t>(
            (uint8_t*)m_rmpack->mmapFile(emb_path), [this, emb_path](uint8_t* p) {
              if (p) {
                m_rmpack->unmapFile(emb_path);
              }
            }
          );
          QNN_INFO("mmap embedding success");
#else
          m_embedding = std::shared_ptr<uint8_t>(
            (uint8_t*)m_rmpack->readFileToMemory(emb_path), [this, emb_path](uint8_t* p) {
              if (p) {
                m_rmpack->freeFileMemory(emb_path);
              }
            }
          );
          QNN_INFO("readFileToMemory embedding success");
#endif
        }
      } else {
          // TODO
          return StatusCode::FAILURE;
          // uint8_t *buffer = (uint8_t*)malloc(file_size);
          // if (!buffer) {
          //     LOG_ERROR("Failed to allocate memory for embedding");
          //     return StatusCode::FAILURE;
          // }
          // emb_file.read(reinterpret_cast<char*>(buffer), file_size);

          // if (emb_file_dtype == QNN_DATATYPE_FLOAT_16) {
          // } else if (emb_file_dtype == QNN_DATATYPE_UFIXED_POINT_16) {
          // } else /* if (emb_file_dtype == QNN_DATATYPE_FLOAT_32) */{
          // }
          // free(buffer);
      }
      if (!is_rmpack) {
        emb_file.close();
      }
    }
  }
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::execute(int token) {
  auto returnStatus = StatusCode::SUCCESS;

  if (!m_tensorsInitialized)
    return StatusCode::FAILURE;

  std::chrono::high_resolution_clock::time_point infer_start = std::chrono::high_resolution_clock::now();
  if (m_embedding == nullptr) {
    int *token_input = (int*)m_ioTensor->getBuffer(&m_inputTensors[0][0]);
    *token_input = token;
  } else {
    void *buffer = m_ioTensor->getBuffer(&m_inputTensors[0][0]);
    int hidden_size = QNN_TENSOR_GET_DIMENSIONS(m_inputTensors[0][0])[QNN_TENSOR_GET_RANK(m_inputTensors[0][0]) - 1];
    if (QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0]) == QNN_DATATYPE_FLOAT_16 || QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0]) == QNN_DATATYPE_UFIXED_POINT_16) {
      memcpy(buffer, m_embedding.get() + token * hidden_size * 2, hidden_size * 2);
    } else if (QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0]) == QNN_DATATYPE_FLOAT_32) {
      memcpy(buffer, m_embedding.get() + token * hidden_size * 4, hidden_size * 4);
    }
  }

  for (int graph_id = 0; graph_id < m_decodeGraphsCount; graph_id++) {
    auto graphInfo     = (*m_decodeGraphsInfo)[graph_id];
    auto executeStatus =
        m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                        m_inputTensors[graph_id],
                                                        graphInfo.numInputTensors,
                                                        m_outputTensors[graph_id],
                                                        graphInfo.numOutputTensors,
                                                        nullptr,
                                                        nullptr);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
    }
  }

  std::chrono::high_resolution_clock::time_point infer_end = std::chrono::high_resolution_clock::now();
  m_lastInferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start);

  return returnStatus;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::executeSequence(std::vector<int> &tokens) {
  auto returnStatus = StatusCode::SUCCESS;


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
    int idx;
    for (idx = 0; (idx+m_prefillSequenceLength) <= tokens.size(); idx += m_prefillSequenceLength) {
      for (int i = 0; i < m_prefillSequenceLength; i++) {
        if (m_embedding != nullptr) {
          void *buffer = m_ioTensor->getBuffer(&m_prefillInputTensors[0][0]);
          int hidden_size = QNN_TENSOR_GET_DIMENSIONS(m_prefillInputTensors[0][0])[QNN_TENSOR_GET_RANK(m_prefillInputTensors[0][0]) - 1];
          if (QNN_TENSOR_GET_DATA_TYPE(m_prefillInputTensors[0][0]) == QNN_DATATYPE_FLOAT_16 || QNN_TENSOR_GET_DATA_TYPE(m_prefillInputTensors[0][0]) == QNN_DATATYPE_UFIXED_POINT_16) {
            memcpy((uint16_t*)buffer + i * hidden_size, m_embedding.get() + tokens[idx + i] * hidden_size * 2, hidden_size * 2);
          } else if (QNN_TENSOR_GET_DATA_TYPE(m_prefillInputTensors[0][0]) == QNN_DATATYPE_FLOAT_32) {
            memcpy((float*)buffer + i * hidden_size, m_embedding.get() + tokens[idx + i] * hidden_size * 4, hidden_size * 4);
          }
        } else {
          int *token_input = (int*)m_ioTensor->getBuffer(&m_prefillInputTensors[0][0]);
          token_input[i] = tokens[idx + i];
        }
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
    std::chrono::high_resolution_clock::time_point infer_end = std::chrono::high_resolution_clock::now();
    m_lastPrefillTime = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start);

    for (; idx < tokens.size(); idx++) {
      if (execute(tokens[idx]) != StatusCode::SUCCESS) {
        QNN_ERROR("Execute failed.");
        return StatusCode::FAILURE;
      }
    }
  }

  if (m_lmhead_weight != nullptr) {
    auto tensor = (Qnn_Tensor_t*)m_logitsOutputTensor;

    void *buffer = m_ioTensor->getBuffer(tensor);
    half_float::half *head_weight = (half_float::half*)m_lmhead_weight.get();

    if (QNN_TENSOR_GET_DATA_TYPE(*tensor) == QNN_DATATYPE_FLOAT_32) {
      for (int i = 0; i < m_vocab_size; i++) {
        m_logitsOutput[i] = 0;
        for (int j = 0; j < m_hidden_size; j++) {
          m_logitsOutput[i] += head_weight[i * m_hidden_size + j] * ((float*)buffer)[j];
        }
      }
    } else if (QNN_TENSOR_GET_DATA_TYPE(*tensor) == QNN_DATATYPE_FLOAT_16) {
        for (int i = 0; i < m_vocab_size; i++) {
          m_logitsOutput[i] = 0;
          for (int j = 0; j < m_hidden_size; j++) {
            m_logitsOutput[i] += head_weight[i * m_hidden_size + j] * ((half_float::half*)buffer)[j];
          }
        }
    } else if (QNN_TENSOR_GET_DATA_TYPE(*tensor) == QNN_DATATYPE_UFIXED_POINT_16) {
        int32_t offset = QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.offset;
        double offsetDouble   = static_cast<double>(offset);
        double scale = QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.scale;
        for (int i = 0; i < m_vocab_size; i++) {
          m_logitsOutput[i] = 0;
          for (int j = 0; j < m_hidden_size; j++) {
            double quantizedValue = static_cast<double>(((uint16_t*)buffer)[j]);
            m_logitsOutput[i] += head_weight[i * m_hidden_size + j] * ((quantizedValue + offsetDouble) * scale);
          }
        }
    }
  }

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
