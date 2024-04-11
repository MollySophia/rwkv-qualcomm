#include "librwkv-qualcomm.h"
#include "librwkv-qualcomm-app.hpp"
#include "DynamicLoadUtil.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnTypeMacros.hpp"
#include "half.hpp"
#include "Logger.hpp"

using namespace qnn::tools;

StatusCode QnnRwkvBackendInitialize(QnnRwkvBackend_t backend, bool context, bool usingHtp) {
    if (!backend) {
        return StatusCode::FAILURE;
    }

    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

    if (rwkv_app::StatusCode::SUCCESS != app->initialize()) {
        QNN_ERROR("Initialization failure");
        return StatusCode::FAILURE;
    }

    if (rwkv_app::StatusCode::SUCCESS != app->initializeBackend()) {
        QNN_ERROR("Backend initialization failure");
        return StatusCode::FAILURE;
    }

    auto devicePropertySupportStatus = app->isDevicePropertySupported();
    if (rwkv_app::StatusCode::FAILURE != devicePropertySupportStatus) {
      auto createDeviceStatus = app->createDevice();
      if (rwkv_app::StatusCode::SUCCESS != createDeviceStatus) {
        QNN_ERROR("Device creation failure");
        return StatusCode::FAILURE;
      }
    }

    if (rwkv_app::StatusCode::SUCCESS != app->initializeProfiling()) {
        QNN_ERROR("Profiling initialization failure");
        return StatusCode::FAILURE;
    }

    if (usingHtp) {
        if (rwkv_app::StatusCode::SUCCESS != app->createPowerConfigId()) {
            QNN_ERROR("Power Config ID creation failure");
        } else {
            if (rwkv_app::StatusCode::SUCCESS != app->setPowerConfig()) {
                QNN_ERROR("Power Config setting failure");
            }
        }
    }

    if (!context) {
        if (rwkv_app::StatusCode::SUCCESS != app->createContext()) {
            QNN_ERROR("Context creation failure");
            return StatusCode::FAILURE;
        }
        if (rwkv_app::StatusCode::SUCCESS != app->composeGraphs()) {
            QNN_ERROR("Graph composition failure");
            return StatusCode::FAILURE;
        }
        if (rwkv_app::StatusCode::SUCCESS != app->finalizeGraphs()) {
            QNN_ERROR("Graph finalization failure");
            return StatusCode::FAILURE;
        }
    } else {
        if (rwkv_app::StatusCode::SUCCESS != app->createFromBinary(app->m_binaryBuffer, app->m_binarySize)) {
            QNN_ERROR("Binary creation failure");
            return StatusCode::FAILURE;
        }
    }

    if (rwkv_app::StatusCode::SUCCESS != app->initializeTensors()) {
        QNN_ERROR("Tensor initialization failure");
        return StatusCode::FAILURE;
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
            rwkv_app::exitWithMessage(
                "Error initializing QNN Function Pointers: could not load backend: " + backendPath,
                EXIT_FAILURE);
        } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
            rwkv_app::exitWithMessage(
                "Error initializing QNN Function Pointers: could not load model: " + modelPath,
                EXIT_FAILURE);
        } else {
            rwkv_app::exitWithMessage("Error initializing QNN Function Pointers", EXIT_FAILURE);
        }
    }

    *backend = new rwkv_app::QnnRwkvApp(qnnFunctionPointers, backendHandle, modelHandle);
    bool usingHtp = backendPath.find("Htp") != std::string::npos;
    return QnnRwkvBackendInitialize(*backend, false, usingHtp);
}

StatusCode QnnRwkvBackendCreateWithContext(
    QnnRwkvBackend_t *backend, QnnRwkvModel_t *modelHandle, std::string contextPath,
    std::string backendPath, std::string systemlibPath
) {
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
            rwkv_app::exitWithMessage(
                "Error initializing QNN Function Pointers: could not load backend: " + backendPath,
                EXIT_FAILURE);
        } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
            rwkv_app::exitWithMessage(
                "Error initializing QNN Function Pointers: could not load model context: " + contextPath,
                EXIT_FAILURE);
        } else {
            rwkv_app::exitWithMessage("Error initializing QNN Function Pointers", EXIT_FAILURE);
        }
    }

    statusCode =
        dynamicloadutil::getQnnSystemFunctionPointers(systemlibPath, &qnnFunctionPointers);
    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
      rwkv_app::exitWithMessage("Error initializing QNN System Function Pointers", EXIT_FAILURE);
    }

    *backend = new rwkv_app::QnnRwkvApp(qnnFunctionPointers, backendHandle, modelHandle, qnn::tools::rwkv_app::ProfilingLevel::OFF,
        contextPath);
    bool usingHtp = backendPath.find("Htp") != std::string::npos;
    return QnnRwkvBackendInitialize(*backend, true, usingHtp);
}

StatusCode QnnRwkvBackendCreateWithContextBuffer(
    QnnRwkvBackend_t *backend, QnnRwkvModel_t *modelHandle, std::string contextPath,
    std::string backendPath, std::string systemlibPath, uint8_t *buffer, uint64_t size) {
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
            rwkv_app::exitWithMessage(
                "Error initializing QNN Function Pointers: could not load backend: " + backendPath,
                EXIT_FAILURE);
        } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
            rwkv_app::exitWithMessage(
                "Error initializing QNN Function Pointers: could not load model context: " + contextPath,
                EXIT_FAILURE);
        } else {
            rwkv_app::exitWithMessage("Error initializing QNN Function Pointers", EXIT_FAILURE);
        }
    }

    statusCode =
        dynamicloadutil::getQnnSystemFunctionPointers(systemlibPath, &qnnFunctionPointers);
    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
      rwkv_app::exitWithMessage("Error initializing QNN System Function Pointers", EXIT_FAILURE);
    }

    *backend = new rwkv_app::QnnRwkvApp(qnnFunctionPointers, backendHandle, modelHandle, qnn::tools::rwkv_app::ProfilingLevel::OFF,
        contextPath);

    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(*backend);
    app->m_binaryBuffer = buffer;
    app->m_binarySize = size;
    bool usingHtp = backendPath.find("Htp") != std::string::npos;
    return QnnRwkvBackendInitialize(*backend, true, usingHtp);
}

StatusCode QnnRwkvSetInput(QnnRwkvBackend_t backend, int inputIdx, float* inputBuffer, size_t inputSize) {
    if (!backend || !inputBuffer) {
        return StatusCode::FAILURE;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

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

    if (QNN_TENSOR_GET_DATA_TYPE(app->m_inputTensors[inputIdx]) == QNN_DATATYPE_FLOAT_32) {
        memcpy(QNN_TENSOR_GET_CLIENT_BUF(app->m_inputTensors[inputIdx]).data, inputBuffer, inputSize * sizeof(float));
    } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_inputTensors[inputIdx]) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)QNN_TENSOR_GET_CLIENT_BUF(app->m_inputTensors[inputIdx]).data;
        for (int i = 0; i < inputSize; i++) {
            ptr[i] = half_float::half(inputBuffer[i]);
        }
    } else {
        app->m_ioTensor.copyFromFloatToNative(inputBuffer, &app->m_inputTensors[inputIdx]);
    }

    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvGetOutput(QnnRwkvBackend_t backend, int outputIdx, float* outputBuffer, size_t outputSize) {
    if (!backend || !outputBuffer) {
        return StatusCode::FAILURE;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);

    std::vector<size_t> shape;
    if (StatusCode::SUCCESS != QnnRwkvGetOutputShape(backend, outputIdx, shape)) {
        return StatusCode::FAILURE;
    }

    int elemcount = 1;
    for (auto dim : shape) {
        elemcount *= dim;
    }

    if (elemcount != outputSize) {
        std::cerr << "Output " << outputIdx << " size mismatch: " << elemcount << " != " << outputSize << std::endl;
        return StatusCode::FAILURE;
    }

    if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[outputIdx]) == QNN_DATATYPE_FLOAT_32) {
        memcpy(outputBuffer, QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[outputIdx]).data, outputSize * sizeof(float));
    } else if (QNN_TENSOR_GET_DATA_TYPE(app->m_outputTensors[outputIdx]) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)QNN_TENSOR_GET_CLIENT_BUF(app->m_outputTensors[outputIdx]).data;
        for (int i = 0; i < outputSize; i++) {
            outputBuffer[i] = ptr[i];
        }
    } else {
        float *buffer;
        app->m_ioTensor.convertToFloat(&buffer, &app->m_outputTensors[outputIdx]);
        memcpy(outputBuffer, buffer, outputSize * sizeof(float));
        free(buffer);
    }

    return StatusCode::SUCCESS;

}

int QnnRwkvGetInputNum(QnnRwkvBackend_t backend) {
    if (!backend) {
        return -1;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    auto graphInfo = (*app->m_graphsInfo)[0];
    return graphInfo.numInputTensors;
}

int QnnRwkvGetOutputNum(QnnRwkvBackend_t backend) {
    if (!backend) {
        return -1;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    auto graphInfo = (*app->m_graphsInfo)[0];
    return graphInfo.numOutputTensors;
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
    for (int i = 0; i < QNN_TENSOR_GET_RANK(app->m_inputTensors[inputIdx]); i++) {
        shape.push_back(*(QNN_TENSOR_GET_DIMENSIONS(app->m_inputTensors[inputIdx]) + i));
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
    for (int i = 0; i < QNN_TENSOR_GET_RANK(app->m_outputTensors[outputIdx]); i++) {
        shape.push_back(*(QNN_TENSOR_GET_DIMENSIONS(app->m_outputTensors[outputIdx]) + i));
    }
    return StatusCode::SUCCESS;

}

StatusCode QnnRwkvExecute(QnnRwkvBackend_t backend, int token) {
    if (!backend) {
        return StatusCode::FAILURE;
    }
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    if (rwkv_app::StatusCode::SUCCESS != app->execute(token)) {
        QNN_ERROR("Execution failure");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvCopyStatesInPlace(QnnRwkvBackend_t backend) {
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    for (size_t idx = 1; idx < (*app->m_graphsInfo)[0].numInputTensors; idx++) {
        app->copyTensor(&app->m_inputTensors[idx], &app->m_outputTensors[idx-1]);
    }

    return StatusCode::SUCCESS;
}

StatusCode QnnRwkvCopyStatesInPlace_v6(QnnRwkvBackend_t backend) {
    rwkv_app::QnnRwkvApp *app = static_cast<rwkv_app::QnnRwkvApp *>(backend);
    for (size_t idx = 0; idx < (*app->m_graphsInfo)[0].numInputTensors/3; idx++) {
        app->copyTensor(&app->m_inputTensors[3*idx+1], &app->m_outputTensors[3*idx]);
        app->copyTensor(&app->m_inputTensors[3*idx+3], &app->m_outputTensors[3*idx+1]);
        app->copyTensor(&app->m_inputTensors[3*idx+2], &app->m_outputTensors[3*idx+2]);
    }

    return StatusCode::SUCCESS;
}