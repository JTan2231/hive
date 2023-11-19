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

void _input_validator(size_t expected, size_t received, const std::string& operation);

void allocateNode(std::shared_ptr<Node> node);

}  // namespace allocation

#endif
