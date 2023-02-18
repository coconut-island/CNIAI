//
// Created by abel on 23-2-15.
//

#include "image_util.h"

#include <iostream>


namespace cniai {
namespace image_util {

int writeBMPi(const char *filename, const unsigned char *chanRGB, int width, int height) {
    unsigned int headers[13];
    FILE *outfile;
    int extra_bytes;
    int padded_size;
    int x;
    int y;
    int n;
    int red, green, blue;

    extra_bytes =
            4 - ((width * 3) % 4);  // How many bytes of padding to add to each
    // horizontal line - the size of which must
    // be a multiple of 4 bytes.
    if (extra_bytes == 4) extra_bytes = 0;

    padded_size = ((width * 3) + extra_bytes) * height;

    // Headers...
    // Note that the "BM" identifier in bytes 0 and 1 is NOT included in these
    // "headers".
    headers[0] = padded_size + 54;  // bfSize (whole file size)
    headers[1] = 0;                // bfReserved (both)
    headers[2] = 54;               // bfOffbits
    headers[3] = 40;               // biSize
    headers[4] = width;            // biWidth
    headers[5] = height;           // biHeight

    // Would have biPlanes and biBitCount in position 6, but they're shorts.
    // It's easier to write them out separately (see below) than pretend
    // they're a single int, especially with endian issues...

    headers[7] = 0;           // biCompression
    headers[8] = padded_size;  // biSizeImage
    headers[9] = 0;           // biXPelsPerMeter
    headers[10] = 0;          // biYPelsPerMeter
    headers[11] = 0;          // biClrUsed
    headers[12] = 0;          // biClrImportant

    if (!(outfile = fopen(filename, "wb"))) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return 1;
    }

    //
    // Headers begin...
    // When printing ints and shorts, we write out 1 character at a time to avoid
    // endian issues.
    //

    fprintf(outfile, "BM");

    for (n = 0; n <= 5; n++) {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int) 0xFF000000) >> 24);
    }

    // These next 4 characters are for the biPlanes and biBitCount fields.

    fprintf(outfile, "%c", 1);
    fprintf(outfile, "%c", 0);
    fprintf(outfile, "%c", 24);
    fprintf(outfile, "%c", 0);

    for (n = 7; n <= 12; n++) {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int) 0xFF000000) >> 24);
    }

    //
    // Headers done, now write the data...
    //
    for (y = height - 1; y >= 0;
         y--)  // BMP image format is written from bottom to top...
    {
        for (x = 0; x <= width - 1; x++) {
            red = chanRGB[(y * width + x) * 3];
            green = chanRGB[(y * width + x) * 3 + 1];
            blue = chanRGB[(y * width + x) * 3 + 2];

            if (red > 255) red = 255;
            if (red < 0) red = 0;
            if (green > 255) green = 255;
            if (green < 0) green = 0;
            if (blue > 255) blue = 255;
            if (blue < 0) blue = 0;
            // Also, it's written in (b,g,r) format...

            fprintf(outfile, "%c", blue);
            fprintf(outfile, "%c", green);
            fprintf(outfile, "%c", red);
        }
        if (extra_bytes)  // See above - BMP lines must be of lengths divisible by 4.
        {
            for (n = 1; n <= extra_bytes; n++) {
                fprintf(outfile, "%c", 0);
            }
        }
    }

    fclose(outfile);
    return 0;
}

}
}