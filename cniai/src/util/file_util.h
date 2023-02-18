//
// Created by abel on 23-2-18.
//

#ifndef CNIAI_TESTS_FILE_UTIL_H
#define CNIAI_TESTS_FILE_UTIL_H

#include <fstream>

namespace cniai {
namespace file_util {

std::ifstream::pos_type filesize(const char* filename);

}
}

#endif //CNIAI_TESTS_FILE_UTIL_H
