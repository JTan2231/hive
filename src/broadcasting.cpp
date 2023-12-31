#include "broadcasting.h"

#include <utility>
#include <vector>

namespace broadcasting {

std::vector<int> padVector(const std::vector<int>& v, int size) {
    std::vector<int> padded = v;
    if (padded.size() < size) {
        std::vector<int> ones(size - padded.size(), 1);
        padded.insert(padded.begin(), ones.begin(), ones.end());
    }

    return padded;
}

std::pair<std::vector<int>, std::vector<int>> padVectors(const std::vector<int>& a, const std::vector<int>& b) {
    int difference = a.size() - b.size();

    if (difference > 0) {
        return {a, padVector(b, a.size())};
    } else if (difference < 0) {
        return {padVector(a, b.size()), b};
    }

    return {a, b};
}

std::pair<std::shared_ptr<BroadcastedBuffer>, std::shared_ptr<BroadcastedBuffer>> makeBroadcastable(
    std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b) {
    auto [a_shape, b_shape] = padVectors(a->shape(), b->shape());

    return {std::shared_ptr<BroadcastedBuffer>(new BroadcastedBuffer(a, a_shape)),
            std::shared_ptr<BroadcastedBuffer>(new BroadcastedBuffer(b, b_shape))};
}

std::vector<int> broadcastedOutputShape(std::shared_ptr<BroadcastedBuffer> a, std::shared_ptr<BroadcastedBuffer> b) {
    if (a->shape().size() != b->shape().size()) {
        std::cerr << strings::error("broadcasting::broadcastedOutputShape error: ")
                  << "shapes must be the same size, got " << strings::info(strings::vecToString(a->shape())) << " and "
                  << strings::info(strings::vecToString(b->shape())) << std::endl;
        exit(-1);
    }

    const std::vector<int>& as = a->shape();
    const std::vector<int>& bs = b->shape();

    std::vector<int> shape;
    for (int i = 0; i < as.size(); i++) {
        shape.push_back(std::max(as[i], bs[i]));
    }

    return shape;
}

bool equivalent(const std::vector<int>& a, const std::vector<int>& b) {
    size_t size = 1;
    for (int i : a) {
        size *= i;
    }

    for (int i : b) {
        size /= i;
    }

    return size == 1;
}

bool broadcastable(std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b) {
    std::vector<int> a = _a->shape();
    std::vector<int> b = _b->shape();

    if (a.size() != b.size()) {
        int padding = std::max(a.size(), b.size());
        a = padVector(a, padding);
        b = padVector(b, padding);
    }

    for (int i = 0; i < a.size(); i++) {
        if (a[i] != b[i] && a[i] != 1 && b[i] != 1) {
            return false;
        }
    }

    return true;
}

std::vector<int> greaterShape(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() > b.size()) {
        return a;
    } else if (b.size() > a.size()) {
        return b;
    }

    for (int i = 0; i < a.size(); i++) {
        if (a[i] > b[i]) {
            return a;
        } else if (b[i] > a[i]) {
            return b;
        }
    }

    return a;
}

}  // namespace broadcasting
