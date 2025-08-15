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

static int sample_logits(const float* logits, const size_t size, float temperature, int top_k, float top_p) {
    // TODO: do sampling directly on qnn tensor
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

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <tokenizer_path> <model_path> <prompt>" << std::endl;
    return EXIT_FAILURE;
  }

  QnnRwkvBackend_t backend;
  QnnRwkvModel_t modelHandle;

  std::string tokenizer_path = argv[1];
  std::string model_path = argv[2];
  std::string prompt_input = argv[3];

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

  trie_tokenizer tokenizer;
  tokenizer.load(tokenizer_path);

  if (model_path.find(".so") != std::string::npos) {
    std::cout << "Loading model lib from " << model_path << std::endl;
    status = QnnRwkvBackendCreate(&backend, &modelHandle, model_path, path + "/libQnnHtp.so");
    if (status != StatusCode::SUCCESS) {
      std::cerr << "QnnRwkvBackendCreate failed" << std::endl;
      return EXIT_FAILURE;
    }
  } else if (model_path.find(".bin") != std::string::npos || model_path.find(".rmpack") != std::string::npos) {
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
  int64_t elemcount = 1;
  for (auto dim : shape) {
    elemcount *= dim;
  }

  std::vector<float> logits(elemcount);

  std::map<int, float> occurences;
  std::vector<double> inference_durations;
  std::string prompt = "User: " + prompt_input + "\n\nAssistant: <think>\n</think>";
  // std::string prompt = "English: " + prompt_input + "\n\nChinese:";
  // std::string prompt = "\n\nThe";
  // "Assistant: 好的，请告诉我诗歌的主题或者一些关键词，这样我才能更好地为您创作一首诗。\n\n"
  // "User: 主题是春天，还有一些关键词可以使用，如花朵、鸟鸣等等。\n\n"
  // "Assistant: 在春天的花园里，\n"
  // "舞动着五彩缤纷的翅膀，\n"
  // "莺啼渐远，笑靥如花，\n"
  // "细雨绵绵，润泽着大地。\n"
  // "这就是春天的景象，\n"
  // "让人心旷神怡，陶醉其中。\n"
  // "愿您在春天里畅游，\n"
  // "欣赏美丽的风景和歌声。\n\n"
  // "User: 生成一个关于夏天的段落。\n\n"
  // "Assistant: 夏天到了！阳光明媚，绿树环绕。沙滩上的海水波澜壮阔，海鸥翱翔。游泳、冲浪、野餐，人们都忙于享受夏日的美好时光。在这个季节里，自然界充满了色彩与生机。草木茂盛，花朵盛开；鸟儿欢快地歌唱着，传递着温暖和喜悦。夏天是一个值得庆祝的季节！\n\n"
  // "User: 谢谢你！\n\n"
  // "Assistant:";
  srand((unsigned)time(NULL));

  const float presence_penalty = 0.4;
  const float freq_penalty = 0.4;
  const float penalty_decay = 0.996;
  const float temperature = 0.7;
  const int top_k = 128;
  const float top_p = 0.9;

  std::vector<int> prompt_ids = tokenizer.Encode(prompt);
  if (QnnRwkvExecuteSequence(backend, prompt_ids) != StatusCode::SUCCESS) {
    std::cerr << "QnnRwkvExecuteSequence failed" << std::endl;
    return EXIT_FAILURE;
  }

  auto duration_prefill = QnnRwkvGetLastPrefillTime(backend);

  QnnRwkvCopyLogitsOutput(backend, logits.data(), logits.size());

  int token = sample_logits(logits.data(), logits.size(), temperature, top_k, top_p);
  std::cout << prompt << "\n============== Prompt End ==============\n";
  std::string output = "";
  for (int i = 0; i < 2000; i++) {
    std::cout << tokenizer.Decode(token);
    output += tokenizer.Decode(token);
    if (QnnRwkvExecute(backend, token) != StatusCode::SUCCESS) {
      std::cerr << "QnnRwkvExecute failed" << std::endl;
      return EXIT_FAILURE;
    }
    QnnRwkvCopyLogitsOutput(backend, logits.data(), logits.size());
    inference_durations.push_back(QnnRwkvGetLastInferenceTime(backend));
    for (auto &x : occurences) {
      logits[x.first] -=
          freq_penalty * x.second + presence_penalty;
      x.second *= penalty_decay;
    }

    token = sample_logits(logits.data(), logits.size(), temperature, top_k, top_p);

    if (token == 0 || (output.size() > 2 && output.substr(output.size() - 2) == "\n\n")) {
      break;
    }

    occurences[token]++;
  }
  std::cout << std::endl;

  double duration_invoke = 0;
  for (auto duration : inference_durations) {
    duration_invoke += duration;
  }

  std::cout << "\n\nTime to first token (" << prompt_ids.size() << " tokens): " << duration_prefill << "s" << std::endl;
  std::cout << "Average tokens per second (prefill): " << (prompt_ids.size() / 16 * 16) / duration_prefill << std::endl;
  std::cout << "Average tokens per second (generation): " << inference_durations.size() / duration_invoke << std::endl;

  QnnRwkvBackendDestroy(backend);
  return EXIT_SUCCESS;
}
