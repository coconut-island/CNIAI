//
// Created by abel on 23-2-15.
//
#ifndef CNIAI_LOG_H
#define CNIAI_LOG_H

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


inline spdlog::level::level_enum getLogLevelEnumFromName(const std::string &levelName){
    if ("trace" == levelName) {
        return spdlog::level::level_enum::trace;
    }

    if ("debug" == levelName) {
        return spdlog::level::level_enum::debug;
    }

    if ("info" == levelName) {
        return spdlog::level::level_enum::info;
    }

    if ("warn" == levelName) {
        return spdlog::level::level_enum::warn;
    }

    if ("err" == levelName || "error" == levelName) {
        return spdlog::level::level_enum::err;
    }

    if ("critical" == levelName) {
        return spdlog::level::level_enum::critical;
    }

    if ("off" == levelName) {
        return spdlog::level::level_enum::off;
    }

    std::cerr << "not match any log levels, level name = " << levelName << std::endl;
    abort();
}


}


#endif //CNIAI_LOG_H
