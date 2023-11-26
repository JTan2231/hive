#include "buffer.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "dtypes.h"
#include "string_utils.h"

Buffer::Buffer(size_t size, DTYPE dtype) : size_(size), dtype_(dtype) {
    data_ = malloc(size * dtypes::dtypeSize(dtype));
    memset(data_, 0, size * dtypes::dtypeSize(dtype));
}

Buffer::~Buffer() {
    free(data_);
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

void Buffer::print() {
    std::cout << strings::debug("Buffer:") << std::endl;
    std::cout << strings::debug("- Size: ") << size_ << std::endl;
}

// TODO:
// these index functions need a second look
// as does the void *
//
// like... really?

// this assumes the entirety of data_ is a 1-D array
// for N-D arrays you'll need to convert
// e.g. 4-D array of shape [x, y, z, w] at [i, j, k, l] means index == i * y * z * w + j * z * w + k * w + l
// TODO: this disclaimer can probably be abstracted more easily
void Buffer::setIndex(size_t index, void* value) {
    if (dtype_ == DTYPE::float32) {
        *(dtypes::toFloat32(data_) + index) = *(dtypes::toFloat32(value));
    }
}

size_t calculateIndex(const std::vector<int>& indices, const std::vector<int>& shape) {
    if (indices.size() != shape.size()) {
        std::cerr << "Node::calculateIndex error: indices.size() must be equal to shape_.size(). Got " << indices.size()
                  << " and " << shape.size() << std::endl;
        exit(-1);
    }

    int shape_prod = 1;
    for (int i : shape) {
        shape_prod *= i;
    }

    size_t final_index = 0;

    for (int i = 0; i < indices.size(); i++) {
        shape_prod /= shape[i];
        final_index += indices[i] * shape_prod;
    }

    return final_index;
}
