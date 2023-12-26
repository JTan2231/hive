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

bool broadcastable(std::vector<int> a, std::vector<int> b) {
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
