#include "logging.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "string_utils.h"

Logging::Logging() {
    std::string filename = log_directory_ + dateTimestamp() + ".log";

    log_file_.open(filename, std::ios::out | std::ios::app);
    if (!log_file_) {
        std::cout << strings::error("Logging::Logging error: ") << "could not open log file " << strings::info(filename)
                  << std::endl;
        exit(-1);
    }
}

Logging::~Logging() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

std::string Logging::dateTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%m-%d-%Y-%H:%M:%S.") << std::setfill('0') << std::setw(6)
       << microseconds.count();

    return ss.str();
}

std::string Logging::timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%H:%M:%S.") << std::setfill('0') << std::setw(6)
       << microseconds.count();

    return ss.str();
}

void Logging::setLogDirectory(std::string directory) {
    log_directory_ = directory;
}

void Logging::errorImpl(const std::string msg) {
    log_file_ << timestamp() << " ERROR: " << msg << std::endl;
}

void Logging::infoImpl(const std::string msg) {
    log_file_ << timestamp() << " INFO: " << msg << std::endl;
}

void Logging::debugImpl(const std::string msg) {
    log_file_ << timestamp() << " DEBUG: " << msg << std::endl;
}
