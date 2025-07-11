#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

class OptimizedTrieTokenizer;

class tokenizer_base {
public:
  tokenizer_base(int pad_token_id, int bos_token_id, int eos_token_id)
      : pad_token_id(pad_token_id), bos_token_id(bos_token_id),
        eos_token_id(eos_token_id) {}
  virtual ~tokenizer_base() = default;
  virtual int load(const std::string vocab_file) = 0;
  virtual std::vector<int> Encode(std::string_view str) const = 0;
  virtual std::string Decode(const std::vector<int> &ids) const = 0;
  virtual std::string Decode(int id) const = 0;
  const int pad_token_id;
  const int bos_token_id;
  const int eos_token_id;
};

class trie_tokenizer : public tokenizer_base {
public:
    trie_tokenizer() : tokenizer_base(0, 0, 0) {};
    int load(const std::string vocab_file);
    std::vector<int> Encode(std::string_view str) const;
    std::string Decode(const std::vector<int> &ids) const;
    std::string Decode(int id) const;
    bool inited() const;
private:
    OptimizedTrieTokenizer * _tokenizer;
};

class abc_tokenizer : public tokenizer_base {
public:
    abc_tokenizer() : tokenizer_base(0, 2, 3) {};
    int load(const std::string) {
        return 0;
    };
    std::vector<int> Encode(std::string_view str) const;
    std::string Decode(const std::vector<int> &ids) const;
    std::string Decode(int id) const;
};

#endif