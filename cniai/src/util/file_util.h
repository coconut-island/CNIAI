//
// Created by abel on 23-2-18.
//

#ifndef CNIAI_FILE_UTIL_H
#define CNIAI_FILE_UTIL_H

#include <fstream>

namespace cniai {
namespace file_util {


std::ifstream::pos_type fileSize(const char* fileName);


}
}


#endif //CNIAI_FILE_UTIL_H
