#ifndef BROADCASTING
#define BROADCASTING

#include <utility>
#include <vector>

#include "buffer.h"

namespace broadcasting {

std::vector<int> padVector(const std::vector<int>& v, int size);

std::pair<std::vector<int>, std::vector<int>> padVectors(const std::vector<int>& a, const std::vector<int>& b);

bool broadcastable(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b);

bool equivalent(const std::vector<int>& a, const std::vector<int>& b);

std::vector<int> broadcastedOutputShape(std::shared_ptr<BroadcastedBuffer> a, std::shared_ptr<BroadcastedBuffer> b);

std::pair<std::shared_ptr<BroadcastedBuffer>, std::shared_ptr<BroadcastedBuffer>> makeBroadcastable(
    std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b);

std::vector<int> greaterShape(const std::vector<int>& a, const std::vector<int>& b);

}  // namespace broadcasting

#endif
