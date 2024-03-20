#include <iostream>
#include <fstream>
#define MSGPACK_NO_BOOST
#include "msgpack.hpp"
#include <chrono>

#include "tokenizer.h"

Tokenizer::Tokenizer(std::string file_path) {
  is_good = false;
  std::ifstream infile;
  infile.open(file_path, std::ios::binary | std::ios::in);
  if (!infile.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return;
  }
  infile.seekg(0, std::ios::end);
  int64_t length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  char *data = new char[length];
  infile.read(data, length);
  infile.close();

  auto unpacker = msgpack::unpack(data, length);
  auto obj = unpacker.get();
  idx2token = obj.as<std::unordered_map<int, std::string>>();
  for (auto &pair : idx2token) {
    token2idx[pair.second] = pair.first;
  }
  delete[] data;
  is_good = true;
}

bool Tokenizer::good() {
  return is_good;
}

std::vector<int> Tokenizer::Encode(std::string str) {
  std::vector<int> ids;
  int str_idx = 0;
  int word_len = 1;
  int id = 0;
  while (str_idx < str.size()) {
    if (str_idx + word_len > str.size()) {
      ids.push_back(id);
      break;
    }
    auto substr = str.substr(str_idx, word_len);
    auto it = token2idx.find(std::string(substr));
    if (it == token2idx.end()) {
      ids.push_back(id);
      str_idx += (word_len - 1);
      word_len = 1;
    } else {
      id = it->second;
      word_len++;
    }
  }

  return ids;
}

std::string Tokenizer::Decode(int id) {
  auto it = idx2token.find(id);
  if (it == idx2token.end()) {
    return "";
  } else {
    return it->second;
  }
}

std::string Tokenizer::Decode(const std::vector<int> &ids) {
  std::string str;
  for (auto id : ids) {
    str += Decode(id);
  }
  return str;
}

std::vector<int> ABCTokenizer::Encode(std::string str) {
  std::vector<int> ids;
  for (int i = 0; i < str.size(); i++) {
    ids.push_back(str[i]);
  }
  return ids;
}

std::string ABCTokenizer::Decode(int id) {
  return std::string(1, id);
}

std::string ABCTokenizer::Decode(const std::vector<int> &ids) {
  std::string str;
  for (auto id : ids) {
    if (id > eos_token_id) {
      str += Decode(id);
    } else {
      str += "";
    }

    if (id == eos_token_id) {
      break;
    }
  }
  return str;
}