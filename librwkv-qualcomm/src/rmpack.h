#ifndef RMPACK_HPP
#define RMPACK_HPP

#include "json.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>

using json = nlohmann::json;

class RMPack {
public:
    struct FileInfo {
        std::string filename;
        uint64_t size;
        uint64_t offset;
    };

    struct MMapInfo {
        void* addr;
        size_t size;
    };

    struct MemoryInfo {
        char* data;
        size_t size;
    };

    explicit RMPack(const std::string& file_path);
    ~RMPack();

    RMPack(const RMPack&) = delete;
    RMPack& operator=(const RMPack&) = delete;

    const json& getConfig() const;
    const std::vector<FileInfo>& getFiles() const;
    const FileInfo* getFileInfo(const std::string& filename) const;
    size_t getFileSize(const std::string& filename) const;
    bool hasFile(const std::string& filename) const;

    void* mmapFile(const std::string& filename);
    void unmapFile(const std::string& filename);

    void* readFileToMemory(const std::string& filename);
    void freeFileMemory(const std::string& filename);

    void listFiles() const;

private:
    static const char* MAGIC_HEADER;
    static const size_t MAGIC_HEADER_SIZE;
    
    std::string file_path_;
    int fd_;
    size_t file_size_;
    json config_;
    std::vector<FileInfo> files_;

    std::map<std::string, MMapInfo> mmap_mappings_;

    std::map<std::string, MemoryInfo> memory_data_;

    void loadFile();
};

#endif // RMPACK_HPP