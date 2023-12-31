#ifndef BUFFER
#define BUFFER

#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

#include "dtypes.h"
#include "string_utils.h"

class Buffer {
   protected:
    // number of elements of type dtype_
    // does NOT represent the number of bits/bytes
    size_t size_;
    void* data_;

    DTYPE dtype_;

   public:
    Buffer();

    Buffer(std::vector<int> shape, DTYPE dtype);

    std::vector<int> shape();

    size_t size();

    DTYPE dtype();

    void* getData();

    void print();

    // this assumes the entirety of data_ is a 1-D array
    // for N-D arrays you'll need to convert
    // e.g. 4-D array of shape [x, y, z, w] at [i, j, k, l] means index == i * y * z * w + j * z * w + k * w + l
    // TODO: this disclaimer can probably be abstracted more easily
    void setIndex(size_t index, void* value);

    void setIndex(const std::vector<int>& index, void* value);

    template <typename T>
    T getIndex(size_t index) {
        if (index > size_) {
            std::cerr << strings::error("Buffer::getIndex error: ") << "index " << strings::info(std::to_string(index))
                      << " out of range " << strings::info(std::to_string(size_)) << std::endl;
            exit(-1);
        }

        return *(((T*)data_) + index);
    }

    template <typename T>
    T getIndex(const std::vector<int>& index) {
        size_t flat_index = 0;
        for (int i = 0; i < index.size(); i++) {
            flat_index += index[i] * strides_[i];
        }

        return getIndex<T>(flat_index);
    }

    std::vector<int> shape_;

    std::vector<int> strides_;
};

class GraphBuffer : public Buffer {
   public:
    GraphBuffer(std::vector<int> shape, DTYPE dtype);
    GraphBuffer(std::shared_ptr<GraphBuffer> buf);
    ~GraphBuffer();
};

class BroadcastedBuffer : public Buffer {
   public:
    BroadcastedBuffer(std::shared_ptr<Buffer> buffer, const std::vector<int>& broadcasted_shape);

    std::vector<int> shape();

   private:
    std::vector<int> broadcasted_shape_;
};

size_t calculateIndex(const std::vector<int>& indices, const std::vector<int>& shape);

#endif
