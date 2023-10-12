#include <cstddef>

#include "dtypes.h"

class Buffer {
    size_t size_;
    void* data_;

    DTYPE dtype_;

   public:
    Buffer(size_t size, DTYPE dtype) : size_(size), dtype_(dtype) {}

    void* getData() { return data_; }
};
