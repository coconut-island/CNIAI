//
// Created by abel on 23-2-15.
//

#include "common/logging.h"
#include <gflags/gflags.h>


DEFINE_string(LOG_LEVEL, "trace", "Log level, includes [trace, debug, info, warn, err, critical, off]");
DEFINE_string(LOG_PATTERN, "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%@] %v", "Log pattern");


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SET_LOG_PATTERN(FLAGS_LOG_PATTERN);
    SET_LOG_LEVEL(FLAGS_LOG_LEVEL);

    LOG_TRACE("{}", "TRACE");
    LOG_DEBUG("{}", "DEBUG");
    LOG_INFO("{}", "INFO");
    LOG_WARN("{}", "WARN");
    LOG_ERROR("{}", "ERROR");
    LOG_CRITICAL("{}", "CRITICAL");

    gflags::ShutDownCommandLineFlags();
    return 0;
}