#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>

#include "librwkv-qualcomm.h"
#include "tokenizer.h"

static int sample_logits(const float* logits, const size_t size, float temperature, int top_k, float top_p) {
    temperature = std::max(temperature, 0.1f);
    temperature = std::min(temperature, 5.f);
    if (top_k >= size)
        top_k = size;

    if (top_k == 0 || top_k == 1)
        return std::max_element(logits, logits + size) - logits;

    // softmax
    float sum = 0;
    int *index = new int[size];
    float *probs = new float[size];

    const float max_logit = *std::max_element(logits, logits + size);

    for (int i = 0; i < size; i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
        index[i] = i;
    }

    if (top_k != size)
        std::nth_element(index, index + top_k,
                index + size,
                [&](int i, int j) { return probs[i] > probs[j]; });
    std::sort(index, index + top_k,
            [&](int i, int j) { return probs[i] > probs[j]; });

    int len = top_k;

    // top-p
    float cumsum = 0;
    for (int i = 0; i < len; i++) {
        probs[index[i]] /= sum;
        cumsum += probs[index[i]];
        if (cumsum >= top_p) {
            len = i + 1;
            break;
        }
    }

    // temperature
    if (fabs(temperature - 1.f) > 1e-6) {
        cumsum = 0;
        for (int i = 0; i < len; i++) {
            probs[index[i]] = std::pow(probs[index[i]], 1.f / temperature);
            cumsum += probs[index[i]];
        }
    }

    // random choice
    float random_value = rand() / float(RAND_MAX) * cumsum;
    
    int ret = -1;
    cumsum = 0;
    for (int i = 0; i < len; i++) {
        cumsum += probs[index[i]];
        if (cumsum >= random_value) {
            ret = index[i];
            break;
        }
    }
    
    delete[] index;
    delete[] probs;
    return ret;
}

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <tokenizer_path> <model_path>" << std::endl;
    return EXIT_FAILURE;
  }

  QnnRwkvBackend_t backend;
  QnnRwkvModel_t modelHandle;

  std::string tokenizer_path = argv[1];
  std::string model_path = argv[2];

  StatusCode status;

  trie_tokenizer tokenizer;
  tokenizer.load(tokenizer_path);

  if (model_path.find(".so") != std::string::npos) {
    std::cout << "Loading model lib from " << model_path << std::endl;
    status = QnnRwkvBackendCreate(&backend, &modelHandle, model_path, "libQnnHtp.so");
    if (status != StatusCode::SUCCESS) {
      std::cerr << "QnnRwkvBackendCreate failed" << std::endl;
      return EXIT_FAILURE;
    }
  } else if (model_path.find(".bin") != std::string::npos) {
    std::cout << "Loading model context binary from " << model_path << std::endl;
    status = QnnRwkvBackendCreateWithContext(&backend, &modelHandle, model_path, "libQnnHtp.so", "libQnnSystem.so");
    if (status != StatusCode::SUCCESS) {
      std::cerr << "QnnRwkvBackendCreateWithContext failed" << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    std::cerr << "Unsupported model file: " << model_path << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<size_t> shape;
  QnnRwkvGetOutputShape(backend, QnnRwkvGetOutputNum(backend) - 1, shape);
  int64_t elemcount = 1;
  for (auto dim : shape) {
    elemcount *= dim;
  }

  std::vector<float> logits(elemcount);

  std::map<int, float> occurences;
  std::vector<double> inference_durations;
  std::string prompt = "User: 请为我写一首诗。\n\nAssistant:";
  srand((unsigned)time(NULL));

  const float presence_penalty = 0.4;
  const float freq_penalty = 0.4;
  const float penalty_decay = 0.996;
  const float temperature = 0.7;
  const int top_k = 128;
  const float top_p = 0.9;

  std::vector<int> prompt_ids = tokenizer.Encode(prompt);
  for (auto token_id : prompt_ids) {
    if (QnnRwkvExecute(backend, token_id) != StatusCode::SUCCESS) {
      std::cerr << "QnnRwkvExecute failed" << std::endl;
      return EXIT_FAILURE;
    }
    inference_durations.push_back(QnnRwkvGetLastInferenceTime(backend));
  }

  QnnRwkvGetOutput(backend, QnnRwkvGetOutputNum(backend) - 1, logits.data(), logits.size());

  int token = sample_logits(logits.data(), logits.size(), temperature, top_k, top_p);
  std::cout << prompt;
  for (int i = 0; i < 300; i++) {
    std::cout << tokenizer.Decode(token);
    if (QnnRwkvExecute(backend, token) != StatusCode::SUCCESS) {
      std::cerr << "QnnRwkvExecute failed" << std::endl;
      return EXIT_FAILURE;
    }
    QnnRwkvGetOutput(backend, QnnRwkvGetOutputNum(backend) - 1, logits.data(), logits.size());
    inference_durations.push_back(QnnRwkvGetLastInferenceTime(backend));
    for (auto &x : occurences) {
      logits[x.first] -=
          freq_penalty * x.second + presence_penalty;
      x.second *= penalty_decay;
    }

    token = sample_logits(logits.data(), logits.size(), temperature, top_k, top_p);

    occurences[token]++;
  }
  std::cout << std::endl;

  double duration_invoke = 0;
  for (auto duration : inference_durations) {
    duration_invoke += duration;
  }
  std::cout << "Average time per token: " << duration_invoke / inference_durations.size() << "s" << std::endl;
  std::cout << "Average tokens per second: " << inference_durations.size() / duration_invoke << std::endl;

  return EXIT_SUCCESS;
}
