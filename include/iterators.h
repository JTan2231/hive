#ifndef ITERATORS
#define ITERATORS

#include <iostream>
#include <vector>

#include "string_utils.h"

namespace iterators {

size_t getFlatIndex(const std::vector<int>& shape, const std::vector<int>& indices);

class IndexIterator {
   public:
    IndexIterator(const std::vector<int>& indices);

    void increment();

    bool end();

    size_t getIndex();

    std::vector<int> getIndices();

    std::vector<int> current_;

    std::vector<int> strides_;

   private:
    std::vector<int> shape_;

    bool end_;
};

class BroadcastIterator {
   public:
    BroadcastIterator(std::vector<int> lesser, std::vector<int> greater);

    bool increment();

    bool decrement();

    void print();

    bool end();

    std::pair<size_t, size_t> getIndices();

    std::vector<int> greater_current_;
    std::vector<int> lesser_current_;

   private:
    size_t getIndex(bool greater);

    void propagateChanges();

    void resetOutOfBounds();

    void updateEnd();

    bool end_;

    std::vector<int> greater_;
    std::vector<int> lesser_;
};

bool lesserGreater(std::vector<int> a, std::vector<int> b);

}  // namespace iterators

#endif
