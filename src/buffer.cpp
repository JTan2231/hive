#include "buffer.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "dtypes.h"

Buffer::Buffer(size_t size, DTYPE dtype) : size_(size), dtype_(dtype) {
    data_ = malloc(size * dtypes::dtypeSize(dtype));
    memset(data_, 0, size * dtypes::dtypeSize(dtype));
}

size_t Buffer::size() {
    return size_;
}
DTYPE Buffer::dtype() {
    return dtype_;
}

void* Buffer::getData() {
    return data_;
}

// this assumes the entirety of data_ is a 1-D array
// for N-D arrays you'll need to convert
// e.g. 4-D array of shape [x, y, z, w] at [i, j, k, l] means index == i * y * z * w + j * z * w + k * w + l
// TODO: this disclaimer can probably be abstracted more easily
void Buffer::setIndex(size_t index, void* value) {
    if (dtype_ == DTYPE::float32) {
        *(dtypes::toFloat32(data_) + index) = *(dtypes::toFloat32(value));
    }
}
