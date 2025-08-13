#include "rmpack.h"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "json.hpp"

using json = nlohmann::json;

const char* RMPack::MAGIC_HEADER = "RWKVMBLE";
const size_t RMPack::MAGIC_HEADER_SIZE = 8;

RMPack::RMPack(const std::string& file_path) : file_path_(file_path), fd_(-1) {
    loadFile();
}

RMPack::~RMPack() {
    for (auto& mapping : mmap_mappings_) {
        if (mapping.second.addr != nullptr) {
            munmap(mapping.second.addr, mapping.second.size);
        }
    }
    for (auto& memory : memory_data_) {
        if (memory.second.data != nullptr) {
            delete[] memory.second.data;
        }
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

void RMPack::loadFile() {
    fd_ = open(file_path_.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("failed to open file: " + file_path_);
    }

    struct stat st;
    if (fstat(fd_, &st) == -1) {
        close(fd_);
        throw std::runtime_error("failed to get file info: " + file_path_);
    }
    file_size_ = st.st_size;

    char header[MAGIC_HEADER_SIZE];
    if (read(fd_, header, MAGIC_HEADER_SIZE) != MAGIC_HEADER_SIZE) {
        close(fd_);
        throw std::runtime_error("failed to read file: " + file_path_);
    }

    if (memcmp(header, MAGIC_HEADER, MAGIC_HEADER_SIZE) != 0) {
        close(fd_);
        throw std::runtime_error("invalid rwkv model file format: " + file_path_);
    }

    uint32_t config_len;
    if (read(fd_, &config_len, sizeof(config_len)) != sizeof(config_len)) {
        close(fd_);
        throw std::runtime_error("failed to read config length");
    }

    std::vector<char> config_buffer(config_len);
    if (read(fd_, config_buffer.data(), config_len) != config_len) {
        close(fd_);
        throw std::runtime_error("failed to read config");
    }
    
    try {
        std::string config_str(config_buffer.begin(), config_buffer.end());
        config_ = json::parse(config_str);
    } catch (const json::exception& e) {
        close(fd_);
        throw std::runtime_error("failed to parse config json: " + std::string(e.what()));
    }

    uint32_t file_count;
    if (read(fd_, &file_count, sizeof(file_count)) != sizeof(file_count)) {
        close(fd_);
        throw std::runtime_error("failed to read file count");
    }

    for (uint32_t i = 0; i < file_count; ++i) {
        FileInfo file_info;

        uint32_t filename_len;
        if (read(fd_, &filename_len, sizeof(filename_len)) != sizeof(filename_len)) {
            close(fd_);
            throw std::runtime_error("failed to read file name length");
        }

        std::vector<char> filename_buffer(filename_len);
        if (read(fd_, filename_buffer.data(), filename_len) != filename_len) {
            close(fd_);
            throw std::runtime_error("failed to read file name");
        }
        file_info.filename = std::string(filename_buffer.begin(), filename_buffer.end());

        if (read(fd_, &file_info.size, sizeof(file_info.size)) != sizeof(file_info.size)) {
            close(fd_);
            throw std::runtime_error("failed to read file size");
        }

        if (read(fd_, &file_info.offset, sizeof(file_info.offset)) != sizeof(file_info.offset)) {
            close(fd_);
            throw std::runtime_error("failed to read file offset");
        }
        
        files_.push_back(file_info);
    }
}

const json& RMPack::getConfig() const {
    return config_;
}

const std::vector<RMPack::FileInfo>& RMPack::getFiles() const {
    return files_;
}

const RMPack::FileInfo* RMPack::getFileInfo(const std::string& filename) const {
    for (const auto& file : files_) {
        if (file.filename == filename) {
            return &file;
        }
    }
    return nullptr;
}

void* RMPack::mmapFile(const std::string& filename) {
    auto it = mmap_mappings_.find(filename);
    if (it != mmap_mappings_.end()) {
        return it->second.addr;
    }

    const FileInfo* file_info = getFileInfo(filename);
    if (!file_info) {
        throw std::runtime_error("file not found: " + filename);
    }

    void* addr = mmap(nullptr, file_info->size, PROT_READ, MAP_PRIVATE, fd_, file_info->offset);
    if (addr == MAP_FAILED) {
        throw std::runtime_error("failed to mmap file: " + filename);
    }

    MMapInfo mmap_info;
    mmap_info.addr = addr;
    mmap_info.size = file_info->size;
    mmap_mappings_[filename] = mmap_info;
    
    return addr;
}

void RMPack::unmapFile(const std::string& filename) {
    auto it = mmap_mappings_.find(filename);
    if (it == mmap_mappings_.end()) {
        return;
    }

    if (it->second.addr != nullptr) {
        munmap(it->second.addr, it->second.size);
    }

    mmap_mappings_.erase(it);
}

void* RMPack::readFileToMemory(const std::string& filename) {
    auto it = memory_data_.find(filename);
    if (it != memory_data_.end()) {
        return it->second.data;
    }

    const FileInfo* file_info = getFileInfo(filename);
    if (!file_info) { 
        throw std::runtime_error("file not found: " + filename);
    }
    
    char* data = new char[file_info->size];

    off_t current_pos = lseek(fd_, 0, SEEK_CUR);

    if (lseek(fd_, file_info->offset, SEEK_SET) == -1) {
        delete[] data;
        throw std::runtime_error("failed to seek file: " + filename);
    }

    ssize_t bytes_read = read(fd_, data, file_info->size);
    if (bytes_read != static_cast<ssize_t>(file_info->size)) {
        delete[] data;
        throw std::runtime_error("failed to read file: " + filename);
    }

    lseek(fd_, current_pos, SEEK_SET);

    MemoryInfo memory_info;
    memory_info.data = data;
    memory_info.size = file_info->size;
    memory_data_[filename] = memory_info;
    
    return data;
}

void RMPack::freeFileMemory(const std::string& filename) {
    auto it = memory_data_.find(filename);
    if (it == memory_data_.end()) {
        return;
    }

    if (it->second.data != nullptr) {
        delete[] it->second.data;
    }

    memory_data_.erase(it);
}

size_t RMPack::getFileSize(const std::string& filename) const {
    const FileInfo* file_info = getFileInfo(filename);
    return file_info ? file_info->size : 0;
}

bool RMPack::hasFile(const std::string& filename) const {
    return getFileInfo(filename) != nullptr;
}

void RMPack::listFiles() const {
    std::cout << "RWKV模型文件: " << file_path_ << std::endl;
    std::cout << "配置项:" << std::endl;
    for (const auto& [key, value] : config_.items()) {
        std::cout << "  " << key << ": " << value << std::endl;
    }
    std::cout << "文件列表:" << std::endl;
    for (const auto& file : files_) {
        std::cout << "  " << file.filename << " (大小: " << file.size 
                  << " 字节, 偏移: " << file.offset << ")" << std::endl;
    }
}
