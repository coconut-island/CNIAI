//
// Created by abel on 23-2-18.
//

#include "file_util.h"

#include <fstream>
#include <iostream>


namespace cniai {
namespace file_util {


std::ifstream::pos_type fileSize(const char* fileName) {
    std::ifstream in(fileName, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}


}
}