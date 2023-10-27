#include "ops.h"

#include <string>
#include <unordered_set>

const std::string Operations::TENSOR = "tensor";
const std::string Operations::MATMUL = "matmul";
const std::string Operations::CONSTANT = "constant";
const std::string Operations::NORMAL = "normal";
const std::unordered_set<std::string> Operations::values_ = {Operations::TENSOR, Operations::MATMUL,
                                                             Operations::CONSTANT, Operations::NORMAL};
