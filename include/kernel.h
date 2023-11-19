#ifndef KERNEL
#define KERNEL

#include <functional>
#include <memory>

#include "graph.h"
#include "ops.h"

namespace kernel {

#define EPSILON 1e-6

// finds and applies the proper kernel to the given node
void computeNode(std::shared_ptr<Node> node);

void _element_wise(
    std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t)>
        element_function,
    std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out);

void _element_wise(std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t)> element_function,
                   std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out);

void _element_wise(
    std::function<void(std::shared_ptr<Buffer>, float, std::shared_ptr<Buffer>, size_t)> element_function,
    std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out);

}  // namespace kernel

#endif
