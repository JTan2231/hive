#ifndef BUFFER
#define BUFFER

#include <cstddef>
#include <cstring>

#include "dtypes.h"

class Buffer {
    // number of elements of type dtype_
    // does NOT represent the number of bits/bytes
    size_t size_;
    void* data_;

    DTYPE dtype_;

   public:
    Buffer(size_t size, DTYPE dtype);

    size_t size();
    DTYPE dtype();

    void* getData();

    // this assumes the entirety of data_ is a 1-D array
    // for N-D arrays you'll need to convert
    // e.g. 4-D array of shape [x, y, z, w] at [i, j, k, l] means index == i * y * z * w + j * z * w + k * w + l
    // TODO: this disclaimer can probably be abstracted more easily
    void setIndex(size_t index, void* value);

    template <typename T>
    T getIndex(size_t index) {
        return *(((T*)data_) + index);
    }
};

#endif
