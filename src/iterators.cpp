#include "iterators.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "string_utils.h"

// these are honestly terrible and need refactored
namespace iterators {

size_t getFlatIndex(const std::vector<int>& shape, const std::vector<int>& indices) {
    if (shape.size() != indices.size()) {
        std::cerr << strings::error("iterators::getFlatIndex error: ")
                  << "shape and indices must be the same size, got " << strings::info(strings::vecToString(shape))
                  << " and " << strings::info(strings::vecToString(indices)) << std::endl;
        exit(-1);
    }

    size_t index = 0;
    size_t stride = 1;
    for (int i = shape.size() - 1; i > -1; i--) {
        index += indices[i] * stride;
        stride *= shape[i];
    }

    return index;
}

IndexIterator::IndexIterator(const std::vector<int>& shape) {
    shape_ = shape;
    current_ = std::vector<int>(shape.size(), 0);

    size_t stride = 1;
    for (int i = shape.size() - 1; i > -1 && strides_.size() < shape.size(); i--) {
        strides_.push_back(stride);
        stride *= shape[i];
    }

    std::reverse(strides_.begin(), strides_.end());

    end_ = false;
}

void IndexIterator::increment() {
    int n = current_.size();

    current_[n - 1]++;

    int change = 0;
    for (int i = n - 1; i > -1; i--) {
        current_[i] += change;

        if (current_[i] >= shape_[i]) {
            if (i == 0) {
                end_ = true;
                return;
            }

            current_[i] = 0;
            change = 1;
        } else {
            change = 0;
        }
    }
}

bool IndexIterator::end() {
    return end_;
}

size_t IndexIterator::getIndex() {
    size_t index = 0;
    for (int i = 0; i < current_.size(); i++) {
        index += current_[i] * strides_[i];
    }

    return index;
}

std::vector<int> IndexIterator::getIndices() {
    return current_;
}

BroadcastIterator::BroadcastIterator(std::vector<int> lesser, std::vector<int> greater) {
    if (lesser.size() != greater.size()) {
        std::cerr << strings::error("BroadcastIterator::BroadcastIterator error: ") << "shape sizes must be equal, got "
                  << strings::info(strings::vecToString(lesser)) << " and "
                  << strings::info(strings::vecToString(greater)) << std::endl;
        exit(-1);
    }

    for (int i = 0; i < lesser.size(); i++) {
        if (lesser[i] != greater[i] && lesser[i] != 1 && greater[i] != 1) {
            std::cerr << strings::error("BroadcastIterator::BroadcastIterator error: ")
                      << "dimensions must be equal or 1, got " << strings::info(strings::vecToString(lesser)) << " and "
                      << strings::info(strings::vecToString(greater)) << std::endl;
            exit(-1);
        }
    }

    lesser_ = lesser;
    greater_ = greater;

    lesser_current_ = std::vector<int>(lesser.size(), 0);
    greater_current_ = std::vector<int>(greater.size(), 0);
}

bool BroadcastIterator::end() {
    return end_;
}

void BroadcastIterator::print() {
    std::cout << "- greater: " << strings::info(strings::vecToString(greater_current_)) << std::endl;
    std::cout << "- lesser: " << strings::info(strings::vecToString(lesser_current_)) << std::endl;
}

// returns {lesser_index, greater_index}
std::pair<size_t, size_t> BroadcastIterator::getIndices() {
    return {getIndex(false), getIndex(true)};
}

size_t BroadcastIterator::getIndex(bool greater) {
    const std::vector<int>& indices = greater ? greater_current_ : lesser_current_;
    const std::vector<int>& shape = greater ? greater_ : lesser_;

    size_t index = indices.back();
    size_t prefix = shape.back();
    for (int i = shape.size() - 2; i > -1; i--) {
        index += indices[i] * prefix;
        prefix *= shape[i];
    }

    return index;
}

void BroadcastIterator::propagateChanges() {
    std::vector<int> change_indices(greater_.size(), 0);
    int change = 0;
    for (int i = greater_current_.size() - 1; i > -1; i--) {
        greater_current_[i] += change;
        change_indices[i] += change;

        if (greater_current_[i] == greater_[i]) {
            change = 1;
        } else if (greater_current_[i] == -1) {
            change = -1;
        } else {
            change = 0;
        }
    }

    for (int i = lesser_current_.size() - 1; i > -1; i--) {
        if (lesser_[i] != 1) {
            lesser_current_[i] += change_indices[i];
        } else if (lesser_current_[i] == 1) {
            lesser_current_[i] = 0;
        }
    }
}

void BroadcastIterator::resetOutOfBounds() {
    for (int i = 0; i < greater_.size(); i++) {
        if (greater_current_[i] == greater_[i]) {
            greater_current_[i] = 0;
        } else if (greater_current_[i] == -1) {
            greater_current_[i] = greater_[i] - 1;
        }

        if (lesser_current_[i] == lesser_[i]) {
            lesser_current_[i] = 0;
        } else if (lesser_current_[i] == -1) {
            lesser_current_[i] = lesser_[i] - 1;
        }
    }
}

void BroadcastIterator::updateEnd() {
    end_ = true;
    for (int i = 0; i < greater_.size(); i++) {
        if (greater_current_[i] != greater_[i] - 1) {
            end_ = false;
            return;
        }
    }
}

bool BroadcastIterator::increment() {
    int n = greater_.size();

    greater_current_[n - 1]++;
    lesser_current_[n - 1]++;
    if (greater_current_[n - 1] == greater_[n - 1]) {
        propagateChanges();
    }

    resetOutOfBounds();
    updateEnd();

    return !end_;
}

bool BroadcastIterator::decrement() {
    int n = greater_.size();

    greater_current_[n - 1]--;
    lesser_current_[n - 1]--;
    if (greater_current_[n - 1] == greater_[n - 1]) {
        propagateChanges();
    }

    resetOutOfBounds();
    updateEnd();

    return !end_;
}

bool lesserGreater(std::vector<int> a, std::vector<int> b) {
    for (int i = 0; i < a.size(); i++) {
        if (a[i] < b[i]) {
            return true;
        }
    }

    return false;
}

}  // namespace iterators
