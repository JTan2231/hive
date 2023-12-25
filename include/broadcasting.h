#ifndef BROADCASTING
#define BROADCASTING

#include <utility>
#include <vector>

namespace broadcasting {

std::vector<int> padVector(const std::vector<int>& v, int size);

std::pair<std::vector<int>, std::vector<int>> padVectors(const std::vector<int>& a, const std::vector<int>& b);

bool broadcastable(std::vector<int> a, std::vector<int> b);

std::vector<int> makeBroadcastable(const std::vector<int>& a, const std::vector<int>& target);

}  // namespace broadcasting

#endif
