#ifndef KERNEL
#define KERNEL

#include <memory>

#include "graph.h"
#include "ops.h"

namespace kernel {

// finds and applies the proper kernel to the given node
void computeNode(std::shared_ptr<Node> node);

}  // namespace kernel

#endif
