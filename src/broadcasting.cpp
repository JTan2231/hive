#include "broadcasting.h"

#include <vector>

namespace broadcasting {

std::vector<int> padVector(const std::vector<int> &v, int size) {
    std::vector<int> padded = v;
    if (padded.size() < size) {
        std::vector<int> ones(size - padded.size(), 1);
        padded.insert(padded.begin(), ones.begin(), ones.end());
    }

    return padded;
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

}  // namespace broadcasting
