#pragma once

#include <string>
#include <vector>

enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAILURE_INPUT_LIST_EXHAUSTED,
  FAILURE_SYSTEM_ERROR,
  FAILURE_SYSTEM_COMMUNICATION_ERROR,
  QNN_FEATURE_UNSUPPORTED
};

typedef void* QnnRwkvBackend_t;

typedef void* QnnRwkvModel_t;

StatusCode QnnRwkvBackendCreate(QnnRwkvBackend_t *backend, QnnRwkvModel_t *modelHandle, std::string modelPath, std::string backendPath);

StatusCode QnnRwkvBackendCreateWithContext(QnnRwkvBackend_t *backend, QnnRwkvModel_t *modelHandle, std::string contextPath, std::string backendPath, std::string systemlibPath);

StatusCode QnnRwkvSetInput(QnnRwkvBackend_t backend, int inputIdx, float* inputBuffer, size_t inputSize);

StatusCode QnnRwkvGetOutput(QnnRwkvBackend_t backend, int outputIdx, float* outputBuffer, size_t outputSize);

int QnnRwkvGetInputNum(QnnRwkvBackend_t backend);

int QnnRwkvGetOutputNum(QnnRwkvBackend_t backend);

StatusCode QnnRwkvGetInputShape(QnnRwkvBackend_t backend, int inputIdx, std::vector<size_t>& shape);

StatusCode QnnRwkvGetOutputShape(QnnRwkvBackend_t backend, int outputIdx, std::vector<size_t>& shape);

StatusCode QnnRwkvExecute(QnnRwkvBackend_t backend, int token);

StatusCode QnnRwkvCopyStatesInPlace(QnnRwkvBackend_t backend);