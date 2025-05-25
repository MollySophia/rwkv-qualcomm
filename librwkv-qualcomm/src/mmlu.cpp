#include <cstdio>
#include <ctime>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <json.hpp>
#include <iostream>
#include "librwkv-qualcomm.h"
#include "tokenizer.h"

using json = nlohmann::json;

struct MMLU_Question {
    std::string prompt;
    std::string answer;
    std::string subject;
};

struct scoreboard {
    int total;
    int correct;
};

int main(int argc, char ** argv) {
    // number of tokens to predict
    int n_predict = 1;

    std::cout.setf(std::ios::unitbuf);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <tokenizer_path> <model_path> <text_path>\n";
        return 1;
    }

    std::string tokenizer_path = argv[1];
    std::string model_path = argv[2];
    std::string text_path = argv[3];

    // load dataset
    std::ifstream file(text_path);
    if (!file.is_open()) {
        throw std::runtime_error("can not load dataset");
    }
    std::string prompt_template = "User: You are a very talented expert in <SUBJECT>. Answer this question:\n<Q>\nA. <|A|>\nB. <|B|>\nC. <|C|>\nD. <|D|>\n\nAssistant: The answer is";

    json data = json::parse(file);
    std::vector<MMLU_Question> questions;
    for (const auto& item : data) {
        std::string prompt = prompt_template;
        prompt = prompt.replace(prompt.find("<SUBJECT>"), 1, item["subject"].get<std::string>());
        prompt = prompt.replace(prompt.find("<Q>"), 1, item["question"].get<std::string>());
        prompt = prompt.replace(prompt.find("<|A|>"), 1, item["choices"][0].get<std::string>());
        prompt = prompt.replace(prompt.find("<|B|>"), 1, item["choices"][1].get<std::string>());
        prompt = prompt.replace(prompt.find("<|C|>"), 1, item["choices"][2].get<std::string>());
        prompt = prompt.replace(prompt.find("<|D|>"), 1, item["choices"][3].get<std::string>());
        MMLU_Question q;
        q.prompt = prompt;
        q.answer = " " + item["answer"].get<std::string>();
        q.subject = item["subject"].get<std::string>();
        questions.push_back(q);
    }

    QnnRwkvBackend_t backend;
    QnnRwkvModel_t modelHandle;

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

    // main loop
    int total_correct = 0;
    std::map<std::string, scoreboard> score_by_subject;
    for (size_t i = 0; i < questions.size(); i++) {
    
        const auto q = questions[i];

        std::string prompt = q.prompt;
        auto prompt_tokens = tokenizer.Encode(prompt);
        prompt_tokens.insert(prompt_tokens.begin(), 0);

        std::string answer;
        QnnRwkvResetStates(backend);
        if (QnnRwkvExecuteSequence(backend, prompt_tokens) != StatusCode::SUCCESS) {
            std::cerr << "QnnRwkvExecuteSequence failed" << std::endl;
            return EXIT_FAILURE;
        }
        QnnRwkvCopyLogitsOutput(backend, output.data(), output.size());
        auto probs = softmax(output);
        auto output_id = std::max_element(probs.begin(), probs.end()) - probs.begin();
        answer = tokenizer.Decode(output_id);
        // printf("Answer: %s\n", answer.c_str());
        // printf("Target: %s\n", q.answer.c_str());

        score_by_subject[q.subject].total++;
        if (answer == q.answer) {
            total_correct++;
            score_by_subject[q.subject].correct++;
        }
        if (i % 10 == 0) { printf("%lu/%lu, correct: %d, acc: %f\n", i+1, questions.size(), total_correct, (float)total_correct/(i+1)); }
    }

    printf("\ncorrect: %d, acc: %f\n", total_correct, (float)total_correct/questions.size());

    json json_output;
    json_output["model"] = model_path;
    json_output["total"] = questions.size();
    json_output["correct"] = total_correct;
    json_output["total_accuracy"] = static_cast<float>(total_correct) / questions.size();
    for (const auto& score : score_by_subject) {
        json_output[score.first] = {
            {"total", score.second.total},
            {"correct", score.second.correct},
            {"accuracy", static_cast<float>(score.second.correct) / score.second.total}
        };
    }

    time_t rawtime;
    struct tm *info;
    char buffer[80];
    time( &rawtime );
    info = localtime( &rawtime );
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", info);
    std::string file_name = "mmlu_results_" + std::string(buffer) + ".json";

    std::ofstream out_file(file_name);
    out_file << json_output.dump(2) << std::endl;

    return 0;
}
