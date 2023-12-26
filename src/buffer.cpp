#include "buffer.h"

#include <emmintrin.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "dtypes.h"
#include "string_utils.h"

// TODO: please god define an iterator for this

Buffer::Buffer(std::vector<int> shape, DTYPE dtype) : shape_(shape), dtype_(dtype) {
    size_t size = 1;
    for (int dim : shape) {
        size *= dim;
    }

    size_ = size;

    data_ = _mm_malloc(size * dtypes::dtypeSize(dtype), 16);
    memset(data_, 0, size * dtypes::dtypeSize(dtype));
}

Buffer::~Buffer() {
    _mm_free(data_);
}

std::vector<int> Buffer::shape() {
    return shape_;
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
    if (index > size_) {
        std::cerr << strings::error("Buffer::setIndex error: ") << "index " << strings::info(std::to_string(index))
                  << " out of range" << std::endl;
        exit(-1);
    }

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
