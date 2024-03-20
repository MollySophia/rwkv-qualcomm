#pragma once
#include <unordered_map>
#include <string>
#include <vector>

// TODO
class TRIE {
public:
    TRIE();

private:
    bool is_end;
    TRIE *next;
};

class Tokenizer {
public:

    Tokenizer(std::string file_path);

    std::vector<int> Encode(std::string str);

    std::string Decode(int id);

    std::string Decode(const std::vector<int> &ids);

    bool good();

private:
    std::unordered_map<int, std::string> idx2token;
    std::unordered_map<std::string, int> token2idx;
    bool is_good;
};

class ABCTokenizer {
public:
    std::vector<int> Encode(std::string str);

    std::string Decode(int id);

    std::string Decode(const std::vector<int> &ids);

    int eos_token_id;
    int bos_token_id;
    int pad_token_id;
};
