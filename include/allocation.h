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

void inputAllocate(std::shared_ptr<Node> node);

// these functions allocate buffers for their given nodes
void tensorAllocate(std::shared_ptr<Node> node);

// NOTE: broadcasting is currently not supported
//       this means given inputs MUST be the same shape,
//       save for the last two dimensions
void matmulAllocate(std::shared_ptr<Node> node);

// all constants will be assumed to be 32-bit float values
void constantAllocate(std::shared_ptr<Node> node);

void normalAllocate(std::shared_ptr<Node> node);

void allocateNode(std::shared_ptr<Node> node);

}  // namespace allocation

#endif
