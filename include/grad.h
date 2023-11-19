#ifndef GRAD
#define GRAD

#include "graph.h"
#include "ops.h"

namespace gradient {

void propagateNode(std::shared_ptr<Node> node);

}  // namespace gradient

#endif
