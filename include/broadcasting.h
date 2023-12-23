#ifndef BROADCASTING
#define BROADCASTING

#include <vector>

namespace broadcasting {

std::vector<int> padVector(const std::vector<int>& v, int size);

bool broadcastable(std::vector<int> a, std::vector<int> b);

}  // namespace broadcasting

#endif
