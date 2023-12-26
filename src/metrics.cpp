#include "metrics.h"

#include <cmath>
#include <iostream>
#include <memory>

#include "buffer.h"

namespace metrics {

Mean::Mean() : sum_(0), count_(0) {
}

float Mean::update(std::shared_ptr<Buffer> buffer) {
    float local_sum = 0;
    for (size_t i = 0; i < buffer->size(); i++) {
        local_sum += buffer->getIndex<float>(i);
    }

    sum_ += local_sum / buffer->size();
    count_++;

    return sum_ / count_;
}

float Mean::value() {
    if (count_ == 0) {
        return 0;
    }

    return sum_ / count_;
}

void Mean::reset() {
    sum_ = 0;
    count_ = 0;
}

MeanAbsoluteError::MeanAbsoluteError() : Mean::Mean() {
}

float MeanAbsoluteError::update(std::shared_ptr<Buffer> pred, std::shared_ptr<Buffer> truth) {
    if (pred->size() != truth->size()) {
        std::cerr << strings::error("metrics::MeanAbsoluteError::update error: ")
                  << "pred and truth must be the same size, got " << strings::info(std::to_string(pred->size()))
                  << " and " << strings::info(std::to_string(truth->size())) << std::endl;
        exit(-1);
    }

    float local_sum = 0;
    for (size_t i = 0; i < pred->size(); i++) {
        local_sum += std::abs(pred->getIndex<float>(i) - truth->getIndex<float>(i));
    }

    local_sum /= pred->size();

    sum_ += local_sum;
    count_++;

    return sum_ / count_;
}

}  // namespace metrics
