//
// Created by abel on 23-2-14.
//

#include "common/logging.h"
#include "common/thread_pool.h"


int main() {
    cniai::ThreadPool threadPool(5);

    for (int i = 0; i < 10; ++i) {
        auto task = [](int thread_idx, int i) {
            LOG_INFO("thread id = {}", thread_idx);
            LOG_INFO("i = {}", i);
            return "res = " + std::to_string(i);
        };
        auto resFuture = threadPool.enqueue(task, i);
        LOG_INFO("{}\n", resFuture.get());
    }

    return 0;
}