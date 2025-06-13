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

StatusCode QnnRwkvGetVocabSize(QnnRwkvBackend_t backend, std::vector<size_t>& shape);

StatusCode QnnRwkvCopyLogitsOutput(QnnRwkvBackend_t backend, float* outputBuffer, size_t outputSize);

StatusCode QnnRwkvExecute(QnnRwkvBackend_t backend, int token);

StatusCode QnnRwkvExecuteSequence(QnnRwkvBackend_t backend, std::vector<int> tokens);

double QnnRwkvGetLastInferenceTime(QnnRwkvBackend_t backend);

double QnnRwkvGetLastPrefillTime(QnnRwkvBackend_t backend);

StatusCode QnnRwkvResetStates(QnnRwkvBackend_t backend);

StatusCode QnnRwkvSaveContext(QnnRwkvBackend_t backend, std::string contextPath);

StatusCode QnnRwkvSetStates(QnnRwkvBackend_t backend, std::vector<std::vector<std::vector<float>>> states);
