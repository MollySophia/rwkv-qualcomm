#include "librwkv-qualcomm.h"
#include "librwkv-qualcomm-app.hpp"
#include "DynamicLoadUtil.hpp"
#include "DataUtil.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnTypeMacros.hpp"
#include "half.hpp"
#include "Logger.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include "soc_detect.h"

#ifdef ANDROID
#include <android/log.h>

#define LOG_ERROR(msg) \
    __android_log_print(ANDROID_LOG_ERROR, "librwkv-qualcomm", "%s", std::string(msg).c_str())

#else
#define LOG_ERROR(msg) \
    std::cout << msg << std::endl
#endif

using namespace qnn::tools;

StatusCode QnnRwkvBackendInitialize(QnnRwkvBackend_t backend, bool context, bool usingHtp, std::string modelPath) {
    if (!backend) {
        return StatusCode::FAILURE;
    }

    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

    if (rwkv_app::StatusCode::SUCCESS != app->initialize()) {
        LOG_ERROR("Initialization failure");
        return StatusCode::FAILURE;
    }

    if (rwkv_app::StatusCode::SUCCESS != app->initializeBackend()) {
        LOG_ERROR("Backend initialization failure");
        return StatusCode::FAILURE;
    }

    auto devicePropertySupportStatus = app->isDevicePropertySupported();
    if (rwkv_app::StatusCode::FAILURE != devicePropertySupportStatus) {
      auto createDeviceStatus = app->createDevice();
      if (rwkv_app::StatusCode::SUCCESS != createDeviceStatus) {
        LOG_ERROR("Device creation failure");
        return StatusCode::FAILURE;
      }
    }

    if (usingHtp) {
        if (rwkv_app::StatusCode::SUCCESS != app->createPowerConfigId()) {
            LOG_ERROR("Power Config ID creation failure");
        } else {
            if (rwkv_app::StatusCode::SUCCESS != app->setRpcLatencyAndPolling()) {
                LOG_ERROR("RpcLatencyAndPolling Config setting failure");
            }

            if (rwkv_app::StatusCode::SUCCESS != app->setPowerConfig()) {
                LOG_ERROR("Power Config setting failure");
            }
        }

        soc_detect detect;
        detect.detect_platform();
        std::string htp_arch = detect.get_htp_arch();
        std::cout << "Htp arch: " << htp_arch << std::endl;

        std::string fullPath;
        if (modelPath.find_last_of("/") != std::string::npos) {
            fullPath = modelPath.substr(0, modelPath.find_last_of("/")) + "/libQnnRwkvWkvOpPackage" + htp_arch + ".so";
        } else {
            fullPath = "libQnnRwkvWkvOpPackage" + htp_arch + ".so";
        }
        std::cout << "Full path: " << fullPath << std::endl;
        std::fstream file(fullPath);
        if (file.good()) {
            LOG_ERROR("Found libQnnRwkvWkvOpPackage.so in LD_LIBRARY_PATH");
            app->m_opPackagePaths.push_back(fullPath + ":RwkvWkvOpPackageInterfaceProvider");

            if (rwkv_app::StatusCode::SUCCESS != app->registerOpPackages()) {
                LOG_ERROR("Op package registration failure");
            }
        }
    }

    if (!context) {
        if (rwkv_app::StatusCode::SUCCESS != app->createContext()) {
            LOG_ERROR("Context creation failure");
            return StatusCode::FAILURE;
        }
        if (rwkv_app::StatusCode::SUCCESS != app->composeGraphs()) {
            LOG_ERROR("Graph composition failure");
            return StatusCode::FAILURE;
        }
        if (rwkv_app::StatusCode::SUCCESS != app->finalizeGraphs()) {
            LOG_ERROR("Graph finalization failure");
            return StatusCode::FAILURE;
        }
    } else {
        if (rwkv_app::StatusCode::SUCCESS != app->createFromBinary(app->m_binaryBuffer, app->m_binarySize)) {
            LOG_ERROR("Binary creation failure");
            return StatusCode::FAILURE;
        }
    }

    if (rwkv_app::StatusCode::SUCCESS != app->initializeTensors()) {
        LOG_ERROR("Tensor initialization failure");
        return StatusCode::FAILURE;
    }

    if (app->m_embedding.empty() && QNN_TENSOR_GET_DATA_TYPE(app->m_inputTensors[0][0]) != QNN_DATATYPE_INT_32) {
        std::string emb_path = modelPath.substr(0, modelPath.find_last_of(".")) + ".emb";
        std::ifstream emb_file;
        emb_file.open(emb_path, std::ios::in|std::ios::binary);
        if (emb_file.is_open()) {
            std::vector<size_t> dims;
            for (int i = 0; i < QNN_TENSOR_GET_RANK(app->m_inputTensors[0][0]); i++) {
                dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(app->m_inputTensors[0][0]) + i));
            }

            int emb_size = dims[dims.size() - 1];
            emb_file.seekg(0, std::ios::end);
            size_t file_size = emb_file.tellg();
            emb_file.seekg(0, std::ios::beg);
            for (int i = 0; i < file_size / (emb_size * sizeof(float)); i++) {
                std::vector<float> emb(emb_size);
                emb_file.read(reinterpret_cast<char*>(emb.data()), emb_size * sizeof(float));
                app->m_embedding.push_back(emb);
            }
            emb_file.close();
        }
    }

    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvBackendCreate(
    QnnRwkvBackend_t *backend, QnnRwkvModel_t *modelHandle, std::string modelPath, std::string backendPath
) {
    if (!qnn::log::initializeLogging()) {
        std::cerr << "ERROR: Unable to initialize logging!\n";
        return StatusCode::FAILURE;
    }
    void* backendHandle;
    rwkv_app::QnnFunctionPointers qnnFunctionPointers;
    auto statusCode = dynamicloadutil::getQnnFunctionPointers(
        backendPath, modelPath, &qnnFunctionPointers, &backendHandle,
        true, modelHandle);

    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
        if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
            LOG_ERROR("Error initializing QNN Function Pointers: could not load backend: " + backendPath);
            return StatusCode::FAILURE;
        } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
            LOG_ERROR("Error initializing QNN Function Pointers: could not load model: " + modelPath);
            return StatusCode::FAILURE;
        } else {
            LOG_ERROR("Error initializing QNN Function Pointers");
            return StatusCode::FAILURE;
        }
    }

    *backend = new rwkv_app::QnnRwkvApp(qnnFunctionPointers, backendHandle, modelHandle);
    bool usingHtp = backendPath.find("Htp") != std::string::npos;
    return QnnRwkvBackendInitialize(*backend, false, usingHtp, modelPath);
}

StatusCode QnnRwkvBackendCreateWithContext(
    QnnRwkvBackend_t *backend, QnnRwkvModel_t *modelHandle, std::string contextPath,
    std::string backendPath, std::string systemlibPath
) {
    if (!qnn::log::initializeLogging()) {
        return StatusCode::FAILURE;
    }
    void* backendHandle;
    rwkv_app::QnnFunctionPointers qnnFunctionPointers;
    auto statusCode = dynamicloadutil::getQnnFunctionPointers(
        backendPath, contextPath, &qnnFunctionPointers, &backendHandle,
        false, modelHandle);

    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
        if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
            LOG_ERROR("Error initializing QNN Function Pointers: could not load backend: " + backendPath);
            return StatusCode::FAILURE;
        } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
            LOG_ERROR("Error initializing QNN Function Pointers: could not load model context: " + contextPath);
            return StatusCode::FAILURE;
        } else {
            LOG_ERROR("Error initializing QNN Function Pointers");
            return StatusCode::FAILURE;
        }
    }

    statusCode =
        dynamicloadutil::getQnnSystemFunctionPointers(systemlibPath, &qnnFunctionPointers);
    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
      LOG_ERROR("Error initializing QNN System Function Pointers");
      return StatusCode::FAILURE;
    }

    *backend = new rwkv_app::QnnRwkvApp(qnnFunctionPointers, backendHandle, modelHandle, std::vector<std::vector<float>>({}),
        contextPath);
    bool usingHtp = backendPath.find("Htp") != std::string::npos;
    return QnnRwkvBackendInitialize(*backend, true, usingHtp, contextPath);
}

StatusCode QnnRwkvCopyLogitsOutput(QnnRwkvBackend_t backend, float* outputBuffer, size_t outputSize) {
    if (!backend || !outputBuffer) {
        return StatusCode::FAILURE;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

    int graph_id = app->m_decodeGraphsCount - 1;
    auto tensor = (Qnn_Tensor_t*)app->m_logitsOutputTensor;

    void *buffer = app->m_ioTensor->getBuffer(tensor);

    if (QNN_TENSOR_GET_DATA_TYPE(*tensor) == QNN_DATATYPE_FLOAT_32) {
        memcpy(outputBuffer, buffer, outputSize * sizeof(float));
    } else if (QNN_TENSOR_GET_DATA_TYPE(*tensor) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)buffer;
        for (int i = 0; i < outputSize; i++) {
            outputBuffer[i] = ptr[i];
        }
    } else if (QNN_TENSOR_GET_DATA_TYPE(*tensor) == QNN_DATATYPE_UFIXED_POINT_16) {
        datautil::tfNToFloat<uint16_t>(outputBuffer, reinterpret_cast<uint16_t*>(buffer),
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.offset,
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.scale,
            outputSize);
    } else {
        LOG_ERROR("Unsupported data type");
        return StatusCode::FAILURE;
    }

    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvGetVocabSize(QnnRwkvBackend_t backend, std::vector<size_t>& shape) {
    if (!backend) {
        return StatusCode::FAILURE;
    }

    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    shape.clear();
    int graph_id = app->m_decodeGraphsCount - 1;
    auto tensor = (Qnn_Tensor_t*)app->m_logitsOutputTensor;

    for (int i = 0; i < QNN_TENSOR_GET_RANK(*tensor); i++) {
        shape.push_back(*(QNN_TENSOR_GET_DIMENSIONS(*tensor) + i));
    }
    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvExecute(QnnRwkvBackend_t backend, int token) {
    if (!backend) {
        return StatusCode::FAILURE;
    }

    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    if (rwkv_app::StatusCode::SUCCESS != app->execute(token)) {
        LOG_ERROR("Execution failure");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvExecuteSequence(QnnRwkvBackend_t backend, std::vector<int> tokens) {
    if (!backend) {
        return StatusCode::FAILURE;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

    if (rwkv_app::StatusCode::SUCCESS != app->executeSequence(tokens)) {
        LOG_ERROR("Execution failure");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

double QnnRwkvGetLastInferenceTime(QnnRwkvBackend_t backend) {
    if (!backend) {
        return -1;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    return app->m_lastInferenceTime.count();
}

StatusCode QnnRwkvResetStates(QnnRwkvBackend_t backend) {
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    if (!app->m_tensorsInitialized)
        return StatusCode::SUCCESS;

    for (size_t graph_id = 0; graph_id < app->m_decodeGraphsCount; graph_id++) {
        for (size_t idx = 0; idx < (*app->m_decodeGraphsInfo)[graph_id].numOutputTensors - 1; idx++) {
            size_t elemcount = 1;
            for (int i = 0; i < QNN_TENSOR_GET_RANK(app->m_outputTensors[graph_id][idx]); i++) {
                elemcount *= *(QNN_TENSOR_GET_DIMENSIONS(app->m_outputTensors[graph_id][idx]) + i);
            }
            void *buffer = app->m_ioTensor->getBuffer(&app->m_outputTensors[graph_id][idx]);
            if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][idx]) == QNN_DATATYPE_FLOAT_16) {
                uint16_t *ptr = (uint16_t*)buffer;
                memset(ptr, 0, elemcount * sizeof(uint16_t));
            } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][idx]) == QNN_DATATYPE_FLOAT_32) {
                float *ptr = (float*)buffer;
                memset(ptr, 0, elemcount * sizeof(float));
            } else {
                app->fillQuantizedTensor(0.0, &app->m_outputTensors[graph_id][idx]);
            }
        }
    }

    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvSaveContext(QnnRwkvBackend_t backend, std::string contextPath) {
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    app->m_outputPath = contextPath;
    app->m_saveBinaryName = "model_cache";
    if (rwkv_app::StatusCode::SUCCESS != app->saveBinary()) {
        LOG_ERROR("Context saving failed");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvSetStates(QnnRwkvBackend_t backend, std::vector<std::vector<std::vector<float>>> states) {
    // rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

    // size_t n_tensors = states[0].size() * states.size();
    // if (n_tensors != app->m_decodeGraphsCount * (app->m_decodeGraphsInfo[0]->numInputTensors - 1)) {
    //     LOG_ERROR("States size mismatch");
    //     return StatusCode::FAILURE;
    // }
    // size_t current_tensor = 0;

    // for (size_t graph_id = 0; graph_id < app->m_decodeGraphsCount; graph_id++) {
    //     for (size_t idx = 0; idx < (*app->m_decodeGraphsInfo)[graph_id].numOutputTensors - 1; idx++) {
    //         size_t states_i = current_tensor / states[0].size();
    //         size_t states_j = current_tensor % states[0].size();
    //         if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][idx]) == QNN_DATATYPE_FLOAT_16) {
    //             uint16_t *ptr = (uint16_t*)QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[graph_id][idx]).data;
    //             for (size_t i = 0; i < states[states_i][states_j].size(); i++) {
    //                 ptr[i] = half_float::half(states[states_i][states_j][i]);
    //             }
    //         } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][idx]) == QNN_DATATYPE_FLOAT_32) {
    //             float *ptr = (float*)QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[graph_id][idx]).data;
    //             for (size_t i = 0; i < states[states_i][states_j].size(); i++) {
    //                 ptr[i] = states[states_i][states_j][i];
    //             }
    //         } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][idx]) == QNN_DATATYPE_UFIXED_POINT_16) {
    //             datautil::floatToTfN<uint16_t>(static_cast<uint16_t*>(QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[graph_id][idx]).data),
    //                 states[states_i][states_j].data(),
    //                 QNN_TENSOR_GET_QUANT_PARAMS(app->m_outputTensors[graph_id][idx]).scaleOffsetEncoding.offset,
    //                 QNN_TENSOR_GET_QUANT_PARAMS(app->m_outputTensors[graph_id][idx]).scaleOffsetEncoding.scale,
    //                 states[states_i][states_j].size());
    //         } else {
    //             LOG_ERROR("Unsupported data type");
    //             return StatusCode::FAILURE;
    //         }
    //         current_tensor++;
    //     }
    // }
    // app->m_inferenced = true;
    // return StatusCode::SUCCESS;
    // TODO
    return StatusCode::SUCCESS;
}