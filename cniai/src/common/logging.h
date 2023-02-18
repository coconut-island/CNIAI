//
// Created by abel on 23-2-15.
//
#ifndef CNIAI_TESTS_CNIAI_LOG_H
#define CNIAI_TESTS_CNIAI_LOG_H

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "spdlog/spdlog.h"

#include <iostream>

#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)

#define SET_LOG_LEVEL(level) spdlog::set_level(cniai::getLogLevelEnumFromName(level))
#define SET_LOG_PATTERN(pattern) spdlog::set_pattern(pattern)

namespace cniai {

inline spdlog::level::level_enum getLogLevelEnumFromName(const std::string &level_name){
    if ("trace" == level_name) {
        return spdlog::level::level_enum::trace;
    }

    if ("debug" == level_name) {
        return spdlog::level::level_enum::debug;
    }

    if ("info" == level_name) {
        return spdlog::level::level_enum::info;
    }

    if ("warn" == level_name) {
        return spdlog::level::level_enum::warn;
    }

    if ("err" == level_name || "error" == level_name) {
        return spdlog::level::level_enum::err;
    }

    if ("critical" == level_name) {
        return spdlog::level::level_enum::critical;
    }

    if ("off" == level_name) {
        return spdlog::level::level_enum::off;
    }

    std::cerr << "not match any log levels, level name = " << level_name << std::endl;
    abort();
}

}


#endif //CNIAI_TESTS_CNIAI_LOG_H
