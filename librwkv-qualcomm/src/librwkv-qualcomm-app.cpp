#include <inttypes.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

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

#include <HTP/QnnHtpPerfInfrastructure.h>
#include <QnnInterface.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>

using namespace qnn;
using namespace qnn::tools;

std::string defaultOutputPath = "./output";

rwkv_app::QnnRwkvApp::QnnRwkvApp(QnnFunctionPointers qnnFunctionPointers,
                                       void* backendLibraryHandle,
                                       void* modelHandle,
                                       std::vector<std::vector<float>> embedding,
                                       rwkv_app::ProfilingLevel profilingLevel,
                                       std::string cachedBinaryPath,
                                       std::string saveBinaryName)
    : m_qnnFunctionPointers(qnnFunctionPointers),
      m_saveBinaryName(saveBinaryName),
      m_cachedBinaryPath(cachedBinaryPath),
      m_profilingLevel(profilingLevel),
      m_backendLibraryHandle(backendLibraryHandle),
      m_modelHandle(modelHandle),
      m_isBackendInitialized(false),
      m_isContextCreated(false) {
  m_embedding = embedding;
  m_outputPath = defaultOutputPath;
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

  // Free Profiling object if it was created
  if (nullptr != m_profileBackendHandle) {
    QNN_DEBUG("Freeing backend profile object.");
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileFree(m_profileBackendHandle)) {
      QNN_ERROR("Could not free backend profile handle.");
    }
  }
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
    QnnLog_Level_t logLevel{QNN_LOG_LEVEL_ERROR};
    log::setLogLevel(logLevel);
    auto logCallback = log::getLogCallback();
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

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::initializeProfiling() {
  if (ProfilingLevel::OFF != m_profilingLevel) {
    QNN_INFO("Profiling turned on; level = %d", m_profilingLevel);
    if (ProfilingLevel::BASIC == m_profilingLevel) {
      QNN_INFO("Basic profiling requested. Creating Qnn Profile object.");
      if (QNN_PROFILE_NO_ERROR !=
          m_qnnFunctionPointers.qnnInterface.profileCreate(
              m_backendHandle, QNN_PROFILE_LEVEL_BASIC, &m_profileBackendHandle)) {
        QNN_WARN("Unable to create profile handle in the backend.");
        return StatusCode::FAILURE;
      }
    } else if (ProfilingLevel::DETAILED == m_profilingLevel) {
      QNN_INFO("Detailed profiling requested. Creating Qnn Profile object.");
      if (QNN_PROFILE_NO_ERROR !=
          m_qnnFunctionPointers.qnnInterface.profileCreate(
              m_backendHandle, QNN_PROFILE_LEVEL_DETAILED, &m_profileBackendHandle)) {
        QNN_ERROR("Unable to create profile handle in the backend.");
        return StatusCode::FAILURE;
      }
    }
  }
  return StatusCode::SUCCESS;
}

// Simple method to report error from app to lib.
int32_t rwkv_app::QnnRwkvApp::reportError(const std::string& err) {
  QNN_ERROR("%s", err.c_str());
  return EXIT_FAILURE;
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
// rwkv_app::StatusCode rwkv_app::QnnRwkvApp::registerOpPackages() {
//   const size_t pathIdx              = 0;
//   const size_t interfaceProviderIdx = 1;
//   for (auto const& opPackagePath : m_opPackagePaths) {
//     std::vector<std::string> opPackage;
//     split(opPackage, opPackagePath, ':');
//     QNN_DEBUG("opPackagePath: %s", opPackagePath.c_str());
//     const char* target     = nullptr;
//     const size_t targetIdx = 2;
//     if (opPackage.size() != 2 && opPackage.size() != 3) {
//       QNN_ERROR("Malformed opPackageString provided: %s", opPackagePath.c_str());
//       return StatusCode::FAILURE;
//     }
//     if (opPackage.size() == 3) {
//       target = (char*)opPackage[targetIdx].c_str();
//     }
//     if (nullptr == m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
//       QNN_ERROR("backendRegisterOpPackageFnHandle is nullptr.");
//       return StatusCode::FAILURE;
//     }
//     if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
//                                     m_backendHandle,
//                                     (char*)opPackage[pathIdx].c_str(),
//                                     (char*)opPackage[interfaceProviderIdx].c_str(),
//                                     target)) {
//       QNN_ERROR("Could not register Op Package: %s and interface provider: %s",
//                 opPackage[pathIdx].c_str(),
//                 opPackage[interfaceProviderIdx].c_str());
//       return StatusCode::FAILURE;
//     }
//     QNN_INFO("Registered Op Package: %s and interface provider: %s",
//              opPackage[pathIdx].c_str(),
//              opPackage[interfaceProviderIdx].c_str());
//   }
//   return StatusCode::SUCCESS;
// }

// Create a Context in a backend.
rwkv_app::StatusCode rwkv_app::QnnRwkvApp::createContext() {
  if (QNN_CONTEXT_NO_ERROR != m_qnnFunctionPointers.qnnInterface.contextCreate(
                                  m_backendHandle,
                                  m_deviceHandle,
                                  (const QnnContext_Config_t**)m_contextConfig,
                                  &m_context)) {
    QNN_ERROR("Could not create context");
    return StatusCode::FAILURE;
  }
  m_isContextCreated = true;
  return StatusCode::SUCCESS;
}

// Free context after done.
rwkv_app::StatusCode rwkv_app::QnnRwkvApp::freeContext() {
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextFree(m_context, m_profileBackendHandle)) {
    QNN_ERROR("Could not free context");
    return StatusCode::FAILURE;
  }
  m_isContextCreated = false;
  return StatusCode::SUCCESS;
}

// Calls composeGraph function in QNN's model.so.
// composeGraphs is supposed to populate graph related
// information in m_graphsInfo and m_graphsCount.
// m_debug is the option supplied to composeGraphs to
// say that all intermediate tensors including output tensors
// are expected to be read by the app.
rwkv_app::StatusCode rwkv_app::QnnRwkvApp::composeGraphs() {
  auto returnStatus = StatusCode::SUCCESS;
  if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR !=
      m_qnnFunctionPointers.composeGraphsFnHandle(
          m_backendHandle,
          m_qnnFunctionPointers.qnnInterface,
          m_context,
          (const qnn_wrapper_api::GraphConfigInfo_t**)m_graphConfigsInfo,
          m_graphConfigsInfoCount,
          &m_graphsInfo,
          &m_graphsCount,
          false,
          log::getLogCallback(),
          log::getLogLevel())) {
    QNN_ERROR("Failed in composeGraphs()");
    returnStatus = StatusCode::FAILURE;
  }
  return returnStatus;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::finalizeGraphs() {
  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    if (QNN_GRAPH_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.graphFinalize(
            (*m_graphsInfo)[graphIdx].graph, m_profileBackendHandle, nullptr)) {
      return StatusCode::FAILURE;
    }
  }
  if (ProfilingLevel::OFF != m_profilingLevel) {
    extractBackendProfilingInfo(m_profileBackendHandle);
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

  std::shared_ptr<uint8_t> buffer{nullptr};

  if (in_buffer && bufferSize) {
    buffer = std::shared_ptr<uint8_t>(in_buffer);
  } else {
    // read serialized binary into a byte buffer
    tools::datautil::StatusCode status{tools::datautil::StatusCode::SUCCESS};
    std::tie(status, bufferSize) = tools::datautil::getFileSize(m_cachedBinaryPath);
    if (0 == bufferSize) {
      QNN_ERROR("Received path to an empty file. Nothing to deserialize.");
      return StatusCode::FAILURE;
    }
    buffer = std::shared_ptr<uint8_t>(new uint8_t[bufferSize], std::default_delete<uint8_t[]>());
    if (!buffer) {
      QNN_ERROR("Failed to allocate memory.");
      return StatusCode::FAILURE;
    }

    status = tools::datautil::readBinaryFromFile(
        m_cachedBinaryPath, reinterpret_cast<uint8_t*>(buffer.get()), bufferSize);
    if (status != tools::datautil::StatusCode::SUCCESS) {
      QNN_ERROR("Failed to read binary data.");
      return StatusCode::FAILURE;
    }
  }

  // inspect binary info
  auto returnStatus = StatusCode::SUCCESS;
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
                         static_cast<void*>(buffer.get()),
                         bufferSize,
                         &binaryInfo,
                         &binaryInfoSize)) {
    QNN_ERROR("Failed to get context binary info");
    returnStatus = StatusCode::FAILURE;
  }

  // fill GraphInfo_t based on binary info
  if (StatusCode::SUCCESS == returnStatus &&
      !copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
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
  if (StatusCode::SUCCESS == returnStatus &&
      m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
          m_backendHandle,
          m_deviceHandle,
          (const QnnContext_Config_t**)m_contextConfig,
          static_cast<void*>(buffer.get()),
          bufferSize,
          &m_context,
          m_profileBackendHandle)) {
    QNN_ERROR("Could not create context from binary.");
    returnStatus = StatusCode::FAILURE;
  }
  if (ProfilingLevel::OFF != m_profilingLevel) {
    extractBackendProfilingInfo(m_profileBackendHandle);
  }
  m_isContextCreated = true;
  if (StatusCode::SUCCESS == returnStatus) {
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
      if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
        QNN_ERROR("graphRetrieveFnHandle is nullptr.");
        returnStatus = StatusCode::FAILURE;
        break;
      }
      if (QNN_SUCCESS !=
          m_qnnFunctionPointers.qnnInterface.graphRetrieve(
              m_context, (*m_graphsInfo)[graphIdx].graphName, &((*m_graphsInfo)[graphIdx].graph))) {
        QNN_ERROR("Unable to retrieve graph handle for graph Idx: %d", graphIdx);
        returnStatus = StatusCode::FAILURE;
      }
    }
  }
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_DEBUG("Cleaning up graph Info structures.");
    qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
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
      m_qnnFunctionPointers.qnnInterface.contextGetBinarySize(m_context, &requiredBufferSize)) {
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
      m_qnnFunctionPointers.qnnInterface.contextGetBinary(m_context,
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

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::extractBackendProfilingInfo(
    Qnn_ProfileHandle_t profileHandle) {
  if (nullptr == m_profileBackendHandle) {
    QNN_ERROR("Backend Profile handle is nullptr; may not be initialized.");
    return StatusCode::FAILURE;
  }
  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetEvents(
                                  profileHandle, &profileEvents, &numEvents)) {
    QNN_ERROR("Failure in profile get events.");
    return StatusCode::FAILURE;
  }
  QNN_DEBUG("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
  for (size_t event = 0; event < numEvents; event++) {
    extractProfilingEvent(*(profileEvents + event));
    extractProfilingSubEvents(*(profileEvents + event));
  }
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::extractProfilingSubEvents(
    QnnProfile_EventId_t profileEventId) {
  const QnnProfile_EventId_t* profileSubEvents{nullptr};
  uint32_t numSubEvents{0};
  if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetSubEvents(
                                  profileEventId, &profileSubEvents, &numSubEvents)) {
    QNN_ERROR("Failure in profile get sub events.");
    return StatusCode::FAILURE;
  }
  QNN_DEBUG("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents, numSubEvents);
  for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
    extractProfilingEvent(*(profileSubEvents + subEvent));
    extractProfilingSubEvents(*(profileSubEvents + subEvent));
  }
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::extractProfilingEvent(
    QnnProfile_EventId_t profileEventId) {
  QnnProfile_EventData_t eventData;
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.profileGetEventData(profileEventId, &eventData)) {
    QNN_ERROR("Failure in profile get event type.");
    return StatusCode::FAILURE;
  }
  QNN_DEBUG("Printing Event Info - Event Type: [%d], Event Value: [%" PRIu64
            "], Event Identifier: [%s], Event Unit: [%d]",
            eventData.type,
            eventData.value,
            eventData.identifier,
            eventData.unit);
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

    // Set power config with different performance parameters
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&powerConfig, NULL};

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

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::initializeTensors() {
  if (nullptr == m_inputTensors[0] || nullptr == m_outputTensors[0]) {
    for (int graph_id = 0; graph_id < m_graphsCount; graph_id++) {
      auto graphInfo     = (*m_graphsInfo)[graph_id];
      // std::cout << "Graph " << graph_id << " : " << graphInfo.graphName << std::endl;
      if (iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&m_inputTensors[graph_id], &m_outputTensors[graph_id], graphInfo)) {
        QNN_ERROR("Error in setting up Input and output Tensors");
        return StatusCode::FAILURE;
      }
      for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
        // std::cout << "Input Tensor " << i << " : " << QNN_TENSOR_GET_NAME(m_inputTensors[graph_id][i]) << " Type: " << QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[graph_id][i]) << std::endl;
        std::vector<size_t> dims;
        for (int j = 0; j < QNN_TENSOR_GET_RANK(m_inputTensors[graph_id][i]); j++) {
          dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(m_inputTensors[graph_id][i]) + j));
        }
        if (QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_16)
          memset(QNN_TENSOR_GET_CLIENT_BUF(m_inputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(uint16_t));
        else if (QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_32)
          memset(QNN_TENSOR_GET_CLIENT_BUF(m_inputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(float));
        else {
          float *ptr = new float[datautil::calculateElementCount(dims)];
          memset(ptr, 0, datautil::calculateElementCount(dims) * sizeof(float));
          m_ioTensor.copyFromFloatToNative(ptr, &m_inputTensors[graph_id][i]);
          delete[] ptr;
        }
      }
      for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
        // std::cout << "Output Tensor " << i << " : " << QNN_TENSOR_GET_NAME(m_outputTensors[graph_id][i]) << " Type: " << QNN_TENSOR_GET_DATA_TYPE(m_outputTensors[graph_id][i]) << std::endl;
        if (std::string(QNN_TENSOR_GET_NAME(m_outputTensors[graph_id][i])).find("kv") != std::string::npos) {
          m_isExternalWkv = true;
        }
        std::vector<size_t> dims;
        for (int j = 0; j < QNN_TENSOR_GET_RANK(m_outputTensors[graph_id][i]); j++) {
          dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(m_outputTensors[graph_id][i]) + j));
        }
        if (QNN_TENSOR_GET_DATA_TYPE(m_outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_16)
          memset(QNN_TENSOR_GET_CLIENT_BUF(m_outputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(uint16_t));
        else if (QNN_TENSOR_GET_DATA_TYPE(m_outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_32)
          memset(QNN_TENSOR_GET_CLIENT_BUF(m_outputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(float));
        else {
          float *ptr = new float[datautil::calculateElementCount(dims)];
          memset(ptr, 0, datautil::calculateElementCount(dims) * sizeof(float));
          m_ioTensor.copyFromFloatToNative(ptr, &m_outputTensors[graph_id][i]);
          delete[] ptr;
        }
      }
    }
  }
  return StatusCode::SUCCESS;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::execute(int token) {
  auto returnStatus = StatusCode::SUCCESS;

  if (nullptr == m_inputTensors[0] || nullptr == m_outputTensors[0])
    return StatusCode::FAILURE;

  if (m_embedding.empty()) {
    int *token_input = (int*)QNN_TENSOR_GET_CLIENT_BUF(m_inputTensors[0][0]).data;
    *token_input = token;
  } else {
    if (QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0]) == QNN_DATATYPE_FLOAT_16) {
      half_float::half *ptr = (half_float::half*)QNN_TENSOR_GET_CLIENT_BUF(m_inputTensors[0][0]).data;
      for (int i = 0; i < m_embedding.size(); i++) {
        ptr[i] = half_float::half(m_embedding[token][i]);
      }
    } else if (QNN_TENSOR_GET_DATA_TYPE(m_inputTensors[0][0]) == QNN_DATATYPE_FLOAT_32) {
      float *ptr = (float*)QNN_TENSOR_GET_CLIENT_BUF(m_inputTensors[0][0]).data;
      memcpy(ptr, m_embedding[token].data(), m_embedding[token].size() * sizeof(float));
    } else {
      m_ioTensor.copyFromFloatToNative(m_embedding[token].data(), &m_inputTensors[0][0]);
    }
  }

  // auto print = [](Qnn_Tensor_t *tensor) {
  //   float *ptr = (float*)QNN_TENSOR_GET_CLIENT_BUF(tensor).data;
  //   for (int i = 0; i < 10; i++) {
  //     std::cout << ptr[i] << " ";
  //   }
  //   std::cout << std::endl;
  // };

  // for (int graph_id = 0; graph_id < m_graphsCount; graph_id++) {
  //   for (int idx = 0; idx < (*m_graphsInfo)[graph_id].numInputTensors; idx++) {
  //     if (idx == 0 && graph_id == 0) continue;
  //     std::cout << "Input Tensor " << idx << " : " << QNN_TENSOR_GET_NAME(m_inputTensors[graph_id][idx]) << std::endl;
  //     print(&m_inputTensors[graph_id][idx]);
  //   }
  // }

  for (int graph_id = 0; graph_id < m_graphsCount; graph_id++) {
    auto graphInfo     = (*m_graphsInfo)[graph_id];
    if (graph_id) { // chunked models
      copyTensor(&m_inputTensors[graph_id][0], &m_outputTensors[graph_id - 1][(*m_graphsInfo)[graph_id - 1].numOutputTensors - 1]);
    }
    auto executeStatus =
        m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                        m_inputTensors[graph_id],
                                                        graphInfo.numInputTensors,
                                                        m_outputTensors[graph_id],
                                                        graphInfo.numOutputTensors,
                                                        m_profileBackendHandle,
                                                        nullptr);
    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
    }
  }

  // for (int graph_id = 0; graph_id < m_graphsCount; graph_id++) {
  //   for (int idx = 0; idx < (*m_graphsInfo)[graph_id].numInputTensors; idx++) {
  //     if (idx == 0 && graph_id == 0) continue;
  //     std::cout << "Input Tensor " << idx << " : " << QNN_TENSOR_GET_NAME(m_inputTensors[graph_id][idx]) << std::endl;
  //     print(&m_inputTensors[graph_id][idx]);
  //   }
  // }

  m_inferenced = true;

  return returnStatus;
}

rwkv_app::StatusCode rwkv_app::QnnRwkvApp::freeGraphs() {
  for (int i = 0; i < m_graphsCount; i++) {
    auto graphInfo     = (*m_graphsInfo)[i];
    m_ioTensor.tearDownInputAndOutputTensors(
        m_inputTensors[i], m_outputTensors[i], graphInfo.numInputTensors, graphInfo.numOutputTensors);
    m_inputTensors[i]  = nullptr;
    m_outputTensors[i] = nullptr;
  }

  qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
  m_graphsInfo = nullptr;
  return StatusCode::SUCCESS;
}

void rwkv_app::QnnRwkvApp::copyTensor(Qnn_Tensor_t *dst, Qnn_Tensor_t *src) {
    std::vector<size_t> dims;
    for (int i = 0; i < QNN_TENSOR_GET_RANK(dst); i++) {
      dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(dst) + i));
    }

    if (QNN_TENSOR_GET_DATA_TYPE(src) == QNN_DATATYPE_FLOAT_16 &&
      QNN_TENSOR_GET_DATA_TYPE(dst) == QNN_DATATYPE_FLOAT_16)
      pal::StringOp::memscpy(QNN_TENSOR_GET_CLIENT_BUF(dst).data,
                            datautil::calculateElementCount(dims) * sizeof(uint16_t),
                            QNN_TENSOR_GET_CLIENT_BUF(src).data,
                            datautil::calculateElementCount(dims) * sizeof(uint16_t));
    else if (QNN_TENSOR_GET_DATA_TYPE(src) == QNN_DATATYPE_FLOAT_32 &&
      QNN_TENSOR_GET_DATA_TYPE(dst) == QNN_DATATYPE_FLOAT_32)
      pal::StringOp::memscpy(QNN_TENSOR_GET_CLIENT_BUF(dst).data,
                            datautil::calculateElementCount(dims) * sizeof(float),
                            QNN_TENSOR_GET_CLIENT_BUF(src).data,
                            datautil::calculateElementCount(dims) * sizeof(float));
    else if (QNN_TENSOR_GET_DATA_TYPE(src) == QNN_DATATYPE_INT_16 &&
      QNN_TENSOR_GET_DATA_TYPE(dst) == QNN_DATATYPE_INT_16) {
      pal::StringOp::memscpy(QNN_TENSOR_GET_CLIENT_BUF(dst).data,
                            datautil::calculateElementCount(dims) * sizeof(int16_t),
                            QNN_TENSOR_GET_CLIENT_BUF(src).data,
                            datautil::calculateElementCount(dims) * sizeof(int16_t));
    }
    else {
      float *buffer;
      if (QNN_TENSOR_GET_DATA_TYPE(src) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)QNN_TENSOR_GET_CLIENT_BUF(src).data;
        buffer = (float*)malloc(datautil::calculateElementCount(dims) * sizeof(float));
        for (int i = 0; i < datautil::calculateElementCount(dims); i++) {
          buffer[i] = float(ptr[i]);
        }
      } else if (QNN_TENSOR_GET_DATA_TYPE(src) != QNN_DATATYPE_FLOAT_32) {
        m_ioTensor.convertToFloat(&buffer, src);
      } else {
        buffer = (float*)QNN_TENSOR_GET_CLIENT_BUF(src).data;
      }

      if (QNN_TENSOR_GET_DATA_TYPE(dst) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)QNN_TENSOR_GET_CLIENT_BUF(dst).data;
        for (int i = 0; i < datautil::calculateElementCount(dims); i++) {
          ptr[i] = half_float::half(buffer[i]);
          free(buffer);
        }
      } else if (QNN_TENSOR_GET_DATA_TYPE(dst) != QNN_DATATYPE_FLOAT_32) {
        m_ioTensor.copyFromFloatToNative(buffer, dst);
        free(buffer);
      } else {
        pal::StringOp::memscpy(QNN_TENSOR_GET_CLIENT_BUF(dst).data,
                              datautil::calculateElementCount(dims) * sizeof(float),
                              buffer,
                              datautil::calculateElementCount(dims) * sizeof(float));
      }
    }
  };
