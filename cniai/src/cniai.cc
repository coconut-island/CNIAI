//
// Created by abel on 23-2-1.
//
#include "common/logging.h"
#include <gflags/gflags.h>

DEFINE_string(log_level, "info", "Log level, includes [trace, debug, info, warn, err, critical, off]");
DEFINE_string(log_pattern, "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%@] %v", "Log pattern");


int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("cniai usage");
    gflags::SetVersionString("0.0.1");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SET_LOG_PATTERN(FLAGS_log_pattern);
    SET_LOG_LEVEL(FLAGS_log_level);

    LOG_INFO("Hello cniai!");

    gflags::ShutDownCommandLineFlags();
    return 0;
}