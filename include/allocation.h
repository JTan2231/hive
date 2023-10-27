#ifndef ALLOCATION
#define ALLOCATION

#include <memory>
#include <vector>

#include "buffer.h"
#include "graph.h"
#include "ops.h"
#include "string_utils.h"

namespace allocation {

// TODO: shapes need figured out to a cleaner solution

// these functions allocate buffers for their given nodes
void allocateTensorNode(std::shared_ptr<Node> node);

// NOTE: broadcasting is currently not supported
//       this means given inputs MUST be the same shape,
//       save for the last two dimensions
void allocateMatmulNode(std::shared_ptr<Node> node);

// all constants will be assumed to be 32-bit float values
void allocateConstantNode(std::shared_ptr<Node> node);

void allocateNormalNode(std::shared_ptr<Node> node);

void allocateNode(std::shared_ptr<Node> node);

}  // namespace allocation

#endif
