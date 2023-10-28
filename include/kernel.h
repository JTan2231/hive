#ifndef KERNEL
#define KERNEL

#include <memory>

#include "graph.h"

namespace kernel {

// finds and applies the proper kernel to the given node
void computeNode(std::shared_ptr<Node> node);

void matmul(std::shared_ptr<Node> node);
void constant(std::shared_ptr<Node> node);

}  // namespace kernel

#endif
