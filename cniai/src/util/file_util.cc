//
// Created by abel on 23-2-18.
//

#include "file_util.h"

#include <fstream>
#include <iostream>


namespace cniai {
namespace file_util {

std::ifstream::pos_type filesize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

}
}