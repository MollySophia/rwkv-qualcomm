#include "librwkv-qualcomm.h"
#include "librwkv-qualcomm-app.hpp"
#include "DynamicLoadUtil.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnTypeMacros.hpp"
#include "half.hpp"
#include "Logger.hpp"
#include <cmath>
#include <fstream>

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

    if (rwkv_app::StatusCode::SUCCESS != app->initializeProfiling()) {
        LOG_ERROR("Profiling initialization failure");
        return StatusCode::FAILURE;
    }

    if (usingHtp) {
        if (rwkv_app::StatusCode::SUCCESS != app->createPowerConfigId()) {
            LOG_ERROR("Power Config ID creation failure");
        } else {
            if (rwkv_app::StatusCode::SUCCESS != app->setPowerConfig()) {
                LOG_ERROR("Power Config setting failure");
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

    if (app->m_embedding.empty()) {
        std::string emb_path = modelPath.substr(0, modelPath.find_last_of(".")) + ".emb";
        std::ifstream emb_file;
        emb_file.open(emb_path, std::ios::in|std::ios::binary);
        if (emb_file.is_open()) {
            std::vector<size_t> dims;
            for (int i = 0; i < QNN_TENSOR_GET_RANK(app->m_inputTensors[0][0]); i++) {
                dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(app->m_inputTensors[0][0]) + i));
            }
            if (dims.size() == 1 && dims[0] != 1) {
                int emb_size = dims[0];
                emb_file.seekg(0, std::ios::end);
                size_t file_size = emb_file.tellg();
                emb_file.seekg(0, std::ios::beg);
                for (int i = 0; i < file_size / (emb_size * sizeof(float)); i++) {
                    std::vector<float> emb(emb_size);
                    emb_file.read(reinterpret_cast<char*>(emb.data()), emb_size * sizeof(float));
                    app->m_embedding.push_back(emb);
                }
            }
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

    *backend = new rwkv_app::QnnRwkvApp(qnnFunctionPointers, backendHandle, modelHandle, std::vector<std::vector<float>>({}), qnn::tools::rwkv_app::ProfilingLevel::OFF,
        contextPath);
    bool usingHtp = backendPath.find("Htp") != std::string::npos;
    return QnnRwkvBackendInitialize(*backend, true, usingHtp, contextPath);
}

StatusCode QnnRwkvBackendCreateWithContextBuffer(
    QnnRwkvBackend_t *backend, QnnRwkvModel_t *modelHandle, std::string contextPath,
    std::string backendPath, std::string systemlibPath, uint8_t *buffer, uint64_t size,
    uint8_t *emb_buffer, uint64_t emb_size, int vocab_size) {
    if (buffer == nullptr || size == 0) {
        return StatusCode::FAILURE;
    }

    if (!qnn::log::initializeLogging()) {
        std::cerr << "ERROR: Unable to initialize logging!\n";
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

    std::vector<std::vector<float>> emb_weight;
    if (emb_buffer != nullptr && emb_size > 0 && vocab_size > 0) {
        auto token_size = emb_size / sizeof(float) / vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            std::vector<float> vec(token_size);
            memcpy(vec.data(), emb_buffer + i * token_size * sizeof(float), token_size * sizeof(float));
            emb_weight.push_back(vec);
        }
    }
    *backend = new rwkv_app::QnnRwkvApp(qnnFunctionPointers, backendHandle, modelHandle, emb_weight, qnn::tools::rwkv_app::ProfilingLevel::OFF,
        contextPath);

    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(*backend);
    app->m_binaryBuffer = buffer;
    app->m_binarySize = size;
    bool usingHtp = backendPath.find("Htp") != std::string::npos;
    return QnnRwkvBackendInitialize(*backend, true, usingHtp, contextPath);
}

StatusCode QnnRwkvSetInput(QnnRwkvBackend_t backend, int inputIdx, float* inputBuffer, size_t inputSize) {
    if (!backend || !inputBuffer) {
        return StatusCode::FAILURE;
    }
    // rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

    std::vector<size_t> shape;
    if (StatusCode::SUCCESS != QnnRwkvGetInputShape(backend, inputIdx, shape)) {
        return StatusCode::FAILURE;
    }

    int elemcount = 1;
    for (auto dim : shape) {
        elemcount *= dim;
    }

    if (elemcount != inputSize) {
        std::cerr << "Input " << inputIdx << " size mismatch: " << elemcount << " != " << inputSize << std::endl;
        return StatusCode::FAILURE;
    }

    // if (QNN_TENSOR_GET_DATA_TYPE(app->m_inputTensors[inputIdx]) == QNN_DATATYPE_FLOAT_32) {
    //     memcpy(QNN_TENSOR_GET_CLIENT_BUF(app->m_inputTensors[inputIdx]).data, inputBuffer, inputSize * sizeof(float));
    // } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_inputTensors[inputIdx]) == QNN_DATATYPE_FLOAT_16) {
    //     half_float::half *ptr = (half_float::half*)QNN_TENSOR_GET_CLIENT_BUF(app->m_inputTensors[inputIdx]).data;
    //     for (int i = 0; i < inputSize; i++) {
    //         ptr[i] = half_float::half(inputBuffer[i]);
    //     }
    // } else {
    //     app->m_ioTensor.copyFromFloatToNative(inputBuffer, &app->m_inputTensors[inputIdx]);
    // }

    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvGetOutput(QnnRwkvBackend_t backend, int outputIdx, float* outputBuffer, size_t outputSize) {
    if (!backend || !outputBuffer) {
        return StatusCode::FAILURE;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

    // std::vector<size_t> shape;
    // if (StatusCode::SUCCESS != QnnRwkvGetOutputShape(backend, outputIdx, shape)) {
    //     return StatusCode::FAILURE;
    // }

    // int elemcount = 1;
    // for (auto dim : shape) {
    //     elemcount *= dim;
    // }

    // if (elemcount != outputSize) {
    //     std::cerr << "Output " << outputIdx << " size mismatch: " << elemcount << " != " << outputSize << std::endl;
    //     return StatusCode::FAILURE;
    // }

    // int output_num = QnnRwkvGetOutputNum(backend);
    // if (outputIdx == output_num - 1) {
        int graph_id = 0, tensor_id = outputIdx;
        if (app->m_graphsCount > 1) {
            graph_id = app->m_graphsCount - 1;
            tensor_id = app->m_graphsInfo[graph_id]->numOutputTensors - 1;
        }

        if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][tensor_id]) == QNN_DATATYPE_FLOAT_32) {
            memcpy(outputBuffer, QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[graph_id][tensor_id]).data, outputSize * sizeof(float));
        } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][tensor_id]) == QNN_DATATYPE_FLOAT_16) {
            half_float::half *ptr = (half_float::half*)QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[graph_id][tensor_id]).data;
            for (int i = 0; i < outputSize; i++) {
                outputBuffer[i] = ptr[i];
            }
        } else {
            float *buffer;
            app->m_ioTensor.convertToFloat(&buffer, &app->m_outputTensors[graph_id][tensor_id]);
            memcpy(outputBuffer, buffer, outputSize * sizeof(float));
            free(buffer);
        }
    // }

    // if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[outputIdx]) == QNN_DATATYPE_FLOAT_32) {
    //     memcpy(outputBuffer, QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[outputIdx]).data, outputSize * sizeof(float));
    // } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[outputIdx]) == QNN_DATATYPE_FLOAT_16) {
    //     half_float::half *ptr = (half_float::half*)QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[outputIdx]).data;
    //     for (int i = 0; i < outputSize; i++) {
    //         outputBuffer[i] = ptr[i];
    //     }
    // } else {
    //     float *buffer;
    //     app->m_ioTensor.convertToFloat(&buffer, &app->m_outputTensors[outputIdx]);
    //     memcpy(outputBuffer, buffer, outputSize * sizeof(float));
    //     free(buffer);
    // }

    return StatusCode::SUCCESS;
}

int QnnRwkvGetInputNum(QnnRwkvBackend_t backend) {
    if (!backend) {
        return -1;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    if (app->m_graphsCount == 1) {
        auto graphInfo = (*app->m_graphsInfo)[0];
        return graphInfo.numInputTensors;
    } else {
        int num = 0;
        for (int i = 0; i < app->m_graphsCount; i++) {
            auto graphInfo = (*app->m_graphsInfo)[i];
            num += graphInfo.numInputTensors;
        }
        return num - app->m_graphsCount + 1;
    }
}

int QnnRwkvGetOutputNum(QnnRwkvBackend_t backend) {
    if (!backend) {
        return -1;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    if (app->m_graphsCount == 1) {
        auto graphInfo = (*app->m_graphsInfo)[0];
        return graphInfo.numOutputTensors;
    } else {
        int num = 0;
        for (int i = 0; i < app->m_graphsCount; i++) {
            auto graphInfo = (*app->m_graphsInfo)[i];
            num += graphInfo.numOutputTensors;
        }
        return num - app->m_graphsCount + 1;
    }
}

StatusCode QnnRwkvGetInputShape(QnnRwkvBackend_t backend, int inputIdx, std::vector<size_t>& shape) {
    if (!backend) {
        return StatusCode::FAILURE;
    }
    if (inputIdx >= QnnRwkvGetInputNum(backend)) {
        std::cerr << "Input index out of bounds" << inputIdx << std::endl;
        return StatusCode::FAILURE;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    shape.clear();
    int graph_id = 0, tensor_id = inputIdx;
    if (app->m_graphsCount > 1) {
        if (inputIdx >= app->m_graphsInfo[0]->numInputTensors) {
            graph_id = (inputIdx - 1) / (app->m_graphsInfo[0]->numInputTensors - 1);
            tensor_id = (inputIdx - 1) % (app->m_graphsInfo[0]->numInputTensors - 1) + 1;
        }
    }

    for (int i = 0; i < QNN_TENSOR_GET_RANK(app->m_inputTensors[graph_id][tensor_id]); i++) {
        shape.push_back(*(QNN_TENSOR_GET_DIMENSIONS(app->m_inputTensors[graph_id][tensor_id]) + i));
    }
    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvGetOutputShape(QnnRwkvBackend_t backend, int outputIdx, std::vector<size_t>& shape) {
    if (!backend) {
        return StatusCode::FAILURE;
    }
    if (outputIdx >= QnnRwkvGetOutputNum(backend)) {
        std::cerr << "Output index out of bounds" << outputIdx << std::endl;
        return StatusCode::FAILURE;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    shape.clear();
    int graph_id = 0, tensor_id = outputIdx;
    if (app->m_graphsCount > 1) {
        if (outputIdx == QnnRwkvGetOutputNum(backend) - 1) {
            graph_id = app->m_graphsCount - 1;
            tensor_id = app->m_graphsInfo[graph_id]->numOutputTensors - 1;
        } else {
            if (outputIdx >= app->m_graphsInfo[0]->numOutputTensors - 1) {
                graph_id = outputIdx / (app->m_graphsInfo[0]->numOutputTensors - 1);
                tensor_id = outputIdx % (app->m_graphsInfo[0]->numOutputTensors - 1);
            }
        }
    }

    for (int i = 0; i < QNN_TENSOR_GET_RANK(app->m_outputTensors[graph_id][tensor_id]); i++) {
        shape.push_back(*(QNN_TENSOR_GET_DIMENSIONS(app->m_outputTensors[graph_id][tensor_id]) + i));
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

StatusCode QnnRwkvCopyStatesInPlace(QnnRwkvBackend_t backend) {
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    if (!app->m_inferenced)
        return StatusCode::SUCCESS;

    for (size_t graph_id = 0; graph_id < app->m_graphsCount; graph_id++) {
        for (size_t idx = 1; idx < (*app->m_graphsInfo)[graph_id].numInputTensors; idx++) {
            // app->copyTensor(&app->m_inputTensors[graph_id][idx], &app->m_outputTensors[graph_id][idx-1]);
            // zero copy
            auto tmp = getQnnTensorClientBuf(app->m_inputTensors[graph_id][idx]);
            setQnnTensorClientBuf(app->m_inputTensors[graph_id][idx], getQnnTensorClientBuf(app->m_outputTensors[graph_id][idx-1]));
            setQnnTensorClientBuf(app->m_outputTensors[graph_id][idx-1], tmp);
        }
    }

    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvResetStates(QnnRwkvBackend_t backend) {
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    if (!app->m_inferenced)
        return StatusCode::SUCCESS;
    for (size_t graph_id = 0; graph_id < app->m_graphsCount; graph_id++) {
        for (size_t idx = 0; idx < (*app->m_graphsInfo)[graph_id].numOutputTensors - 1; idx++) {
            size_t elemcount = 1;
            for (int i = 0; i < QNN_TENSOR_GET_RANK(app->m_outputTensors[graph_id][idx]); i++) {
                elemcount *= *(QNN_TENSOR_GET_DIMENSIONS(app->m_outputTensors[graph_id][idx]) + i);
            }
            if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][idx]) == QNN_DATATYPE_FLOAT_16) {
                uint16_t *ptr = (uint16_t*)QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[graph_id][idx]).data;
                memset(ptr, 0, elemcount * sizeof(uint16_t));
            } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][idx]) == QNN_DATATYPE_FLOAT_32) {
                float *ptr = (float*)QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[graph_id][idx]).data;
                memset(ptr, 0, elemcount * sizeof(float));
            } else {
                // TODO: optimize
                float *ptr = new float[elemcount];
                memset(ptr, 0, elemcount * sizeof(float));
                app->m_ioTensor.copyFromFloatToNative(ptr, &app->m_outputTensors[graph_id][idx]);
                delete[] ptr;
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
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

    size_t n_tensors = states[0].size() * states.size();
    if (n_tensors != app->m_graphsCount * (app->m_graphsInfo[0]->numInputTensors - 1)) {
        LOG_ERROR("States size mismatch");
        return StatusCode::FAILURE;
    }
    size_t current_tensor = 0;

    for (size_t graph_id = 0; graph_id < app->m_graphsCount; graph_id++) {
        for (size_t idx = 0; idx < (*app->m_graphsInfo)[graph_id].numOutputTensors - 1; idx++) {
            size_t states_i = current_tensor / states[0].size();
            size_t states_j = current_tensor % states[0].size();
            if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][idx]) == QNN_DATATYPE_FLOAT_16) {
                uint16_t *ptr = (uint16_t*)QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[graph_id][idx]).data;
                for (size_t i = 0; i < states[states_i][states_j].size(); i++) {
                    ptr[i] = half_float::half(states[states_i][states_j][i] / 8);
                }
            } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[graph_id][idx]) == QNN_DATATYPE_FLOAT_32) {
                float *ptr = (float*)QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[graph_id][idx]).data;
                for (size_t i = 0; i < states[states_i][states_j].size(); i++) {
                    ptr[i] = states[states_i][states_j][i] / 8;
                }
            } else {
                // TODO: optimize
                float *ptr = new float[states[states_i][states_j].size()];
                for (size_t i = 0; i < states[states_i][states_j].size(); i++) {
                    ptr[i] = states[states_i][states_j][i] / 8;
                }
                app->m_ioTensor.copyFromFloatToNative(ptr, &app->m_outputTensors[graph_id][idx]);
                delete[] ptr;
            }
            current_tensor++;
        }
    }
    app->m_inferenced = true;
    return StatusCode::SUCCESS;
}