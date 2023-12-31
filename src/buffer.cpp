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

Buffer::Buffer() {
}

Buffer::Buffer(std::vector<int> shape, DTYPE dtype) : shape_(shape), dtype_(dtype) {
}

GraphBuffer::GraphBuffer(std::vector<int> shape, DTYPE dtype) : Buffer(shape, dtype) {
    size_t size = 1;
    for (int dim : shape) {
        size *= dim;
    }

    size_ = size;

    data_ = _mm_malloc(size * dtypes::dtypeSize(dtype), 16);
    memset(data_, 0, size * dtypes::dtypeSize(dtype));

    strides_ = std::vector<int>(shape.size(), 0);

    size_t stride = 1;
    for (int i = shape.size() - 1; i > -1; i--) {
        if (shape[i] == 1) {
            strides_[i] = 0;
        } else {
            strides_[i] = stride;
        }

        stride *= shape[i];
    }
}

GraphBuffer::GraphBuffer(std::shared_ptr<GraphBuffer> buf) {
    size_ = buf->size_;
    data_ = buf->data_;
    shape_ = buf->shape_;
    dtype_ = buf->dtype_;
    strides_ = buf->strides_;
}

GraphBuffer::~GraphBuffer() {
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

void Buffer::setIndex(const std::vector<int>& index, void* value) {
    size_t flat_index = 0;
    for (int i = 0; i < index.size(); i++) {
        flat_index += index[i] * strides_[i];
    }

    setIndex(flat_index, value);
}

BroadcastedBuffer::BroadcastedBuffer(std::shared_ptr<Buffer> buffer, const std::vector<int>& broadcasted_shape)
    : broadcasted_shape_(broadcasted_shape) {
    // recalculate `strides_` with `broadcasted_shape`
    size_ = buffer->size();
    data_ = buffer->getData();
    shape_ = buffer->shape_;
    dtype_ = buffer->dtype();
    strides_ = std::vector<int>(broadcasted_shape_.size(), 0);
    size_t stride = 1;
    for (int i = broadcasted_shape.size() - 1; i > -1; i--) {
        if (broadcasted_shape[i] == 1) {
            strides_[i] = 0;
        } else {
            strides_[i] = stride;
        }

        stride *= broadcasted_shape[i];
    }
}

std::vector<int> BroadcastedBuffer::shape() {
    return broadcasted_shape_;
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
