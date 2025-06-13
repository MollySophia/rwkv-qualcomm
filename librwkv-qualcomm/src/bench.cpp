#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
#include <stdlib.h>
#include <unistd.h>

#include "librwkv-qualcomm.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
    return EXIT_FAILURE;
  }

  QnnRwkvBackend_t backend;
  QnnRwkvModel_t modelHandle;

  std::string model_path = argv[1];

  char *buffer;
  if ((buffer = getcwd(NULL, 0)) == NULL) {
    perror("getcwd error");
  }
  std::string path = std::string(buffer);
  setenv("LD_LIBRARY_PATH", path.c_str(), 1);
  setenv("ADSP_LIBRARY_PATH", path.c_str(), 1);
  if (buffer) {
    free(buffer);
  }
  std::cout << "cwd: " << path << std::endl;

  StatusCode status;

  if (model_path.find(".so") != std::string::npos) {
    std::cout << "Loading model lib from " << model_path << std::endl;
    status = QnnRwkvBackendCreate(&backend, &modelHandle, model_path, path + "/libQnnHtp.so");
    if (status != StatusCode::SUCCESS) {
      std::cerr << "QnnRwkvBackendCreate failed" << std::endl;
      return EXIT_FAILURE;
    }
  } else if (model_path.find(".bin") != std::string::npos) {
    std::cout << "Loading model context binary from " << model_path << std::endl;
    status = QnnRwkvBackendCreateWithContext(&backend, &modelHandle, model_path, path + "/libQnnHtp.so", path + "/libQnnSystem.so");
    if (status != StatusCode::SUCCESS) {
      std::cerr << "QnnRwkvBackendCreateWithContext failed" << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    std::cerr << "Unsupported model file: " << model_path << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<size_t> shape;
  QnnRwkvGetVocabSize(backend, shape);
  int64_t vocab_size = 1;
  for (auto dim : shape) {
    vocab_size *= dim;
  }

  std::vector<float> logits(vocab_size);

  std::map<int, float> occurences;
  std::vector<double> inference_durations;
 
  srand((unsigned)time(NULL));

  const float presence_penalty = 0.4;
  const float freq_penalty = 0.4;
  const float penalty_decay = 0.996;
  const float temperature = 0.7;
  const int top_k = 128;
  const float top_p = 0.9;

  std::vector<int> prompt_ids;
  for (int i = 0; i < 512; i++) {
    prompt_ids.push_back(rand() % vocab_size);
  }
  if (QnnRwkvExecuteSequence(backend, prompt_ids) != StatusCode::SUCCESS) {
    std::cerr << "QnnRwkvExecuteSequence failed" << std::endl;
    return EXIT_FAILURE;
  }
  auto duration_prefill = QnnRwkvGetLastInferenceTime(backend);

  // QnnRwkvCopyLogitsOutput(backend, logits.data(), logits.size());
  for (int i = 0; i < 512; i++) {
    int token = rand() % vocab_size;
    if (QnnRwkvExecute(backend, token) != StatusCode::SUCCESS) {
      std::cerr << "QnnRwkvExecute failed" << std::endl;
      return EXIT_FAILURE;
    }
    // QnnRwkvCopyLogitsOutput(backend, logits.data(), logits.size());
    inference_durations.push_back(QnnRwkvGetLastInferenceTime(backend));
  }

  double duration_invoke = 0;
  for (auto duration : inference_durations) {
    duration_invoke += duration;
  }

  std::cout << "\n\nTime to first token (" << prompt_ids.size() << " tokens): " << duration_prefill << "s" << std::endl;
  std::cout << "Average tokens per second (prefill): " << prompt_ids.size() / duration_prefill << std::endl;
  std::cout << "Average tokens per second (generation): " << inference_durations.size() / duration_invoke << std::endl;

  return EXIT_SUCCESS;
}
