#ifndef GRAD
#define GRAD

#include "graph.h"
#include "ops.h"

namespace gradient {

void _propagate_current_grad(std::shared_ptr<Node> node, std::shared_ptr<Node> child);

void propagateNode(std::shared_ptr<Node> node);

}  // namespace gradient

#endif
