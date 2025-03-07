#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "librwkv-qualcomm.h"
#include "tokenizer.h"

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <tokenizer_path> <model_path> <text_path>\n";
        return 1;
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
    QnnRwkvGetVocabSize(backend, shape);
    int64_t elemcount = 1;
    for (auto dim : shape) {
        elemcount *= dim;
    }
    std::cout << "Output elemcount:" << elemcount << std::endl;

    std::vector<float> output(elemcount);

    char *eval_text_buf;
    std::ifstream eval_text_file(argv[3], std::ios::binary | std::ios::ate);
    size_t file_size;
    if (eval_text_file.is_open()) {
        eval_text_file.seekg(0, std::ios::end);
        file_size = eval_text_file.tellg();
        eval_text_buf = new char[file_size];
        eval_text_file.seekg(0, std::ios::beg);
        eval_text_file.read(eval_text_buf, file_size);
        eval_text_file.close();
    } else {
        std::cerr << "Unable to open file\n";
        return 1;
    }
    std::vector<std::string> eval_text;
    size_t next = 0;
    for (size_t i = 0; i < file_size; i++) {
        if (eval_text_buf[i] == '|') {
            eval_text.push_back(std::string(eval_text_buf + next, i - next));
            next = i + 1;
        }
    }
    delete[] eval_text_buf;
    std::cout << "Eval texts num: " << eval_text.size() << std::endl;

    float xsum = 0;
    int xcnt = 0;
    int xacc = 0;

    auto softmax = [](std::vector<float> &logits) {
        std::vector<float> probs(logits.size());
        float max_val = *std::max_element(logits.begin(), logits.end());
        float sum = 0;
        for (size_t i = 0; i < logits.size(); i++) {
            probs[i] = std::exp((logits[i] - max_val));
            sum += probs[i];
        }
        for (size_t i = 0; i < logits.size(); i++) {
            probs[i] /= sum;
        }
        return probs;
    };

    for (const auto &text : eval_text) {
        std::cout << "Sample num: " << xcnt << std::endl;
        auto prompt_ids = tokenizer.Encode(text.substr(0, text.find_last_of(' ')));
        prompt_ids.insert(prompt_ids.begin(), 0);
        auto target_ids = tokenizer.Encode(text.substr(text.find_last_of(' ')));
        std::cout << "Prompt: " << text.substr(0, text.find_last_of(' ')) << std::endl;
        std::cout << "Target: " << text.substr(text.find_last_of(' ')) << std::endl;
        QnnRwkvResetStates(backend);
        std::cout << "Response: ";

        bool correct = true;
        float logits_val = 0;
        if (QnnRwkvExecuteSequence(backend, prompt_ids) != StatusCode::SUCCESS) {
            std::cerr << "QnnRwkvExecuteSequence failed" << std::endl;
            return EXIT_FAILURE;
        }

        QnnRwkvCopyLogitsOutput(backend, output.data(), output.size());
        auto probs = softmax(output);
        for (int i = 0; i < target_ids.size(); i++) {
            auto output_id = std::max_element(probs.begin(), probs.end()) - probs.begin();
            logits_val += std::log(probs[target_ids[i]]);
            if (output_id != target_ids[i]) {
                correct = false;
            }
            std::cout << tokenizer.Decode(output_id);

            if (QnnRwkvExecute(backend, target_ids[i]) != StatusCode::SUCCESS) {
                std::cerr << "QnnRwkvExecute failed" << std::endl;
                return EXIT_FAILURE;
            }
            QnnRwkvCopyLogitsOutput(backend, output.data(), output.size());
            probs = softmax(output);
        }

        xcnt++;
        if (correct) {
            xacc++;
        } 
        xsum += logits_val;

        // if (xcnt % 10 == 0) {
            std::cout << "\nAccuracy: " << xacc << "/" << xcnt << " = " << (float)xacc / xcnt << std::endl;
            std::cout << "Perplexity: " << std::exp(-xsum / xcnt) << std::endl;
            std::cout << "====================================\n";
        // }
    }

    
    return 0;
}