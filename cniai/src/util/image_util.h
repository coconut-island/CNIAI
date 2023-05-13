//
// Created by abel on 23-2-15.
//

#ifndef CNIAI_IMAGE_UTIL_H
#define CNIAI_IMAGE_UTIL_H


namespace cniai {
namespace image_util {


int writeBMPi(const char *fileName, const unsigned char *chanRGB, int width, int height);

int writeYU12(const char *fileName, const unsigned char *chanYU12, int width, int height);


}
}


#endif //CNIAI_IMAGE_UTIL_H
