#include "grad.h"

#include <memory>

#include "buffer.h"
#include "buffer_ops.h"
#include "graph.h"
#include "ops.h"
#include "string_utils.h"

// TODO: implement the unimplemented

namespace gradient {

void propagateNode(std::shared_ptr<Node> node) {
    const auto& gradMap = OperationRegistry::GetGradientMap();

    auto it = gradMap.find(node->operation_type_);
    if (it != gradMap.end()) {
        it->second(node);
    } else {
        std::cerr << strings::error("gradient::propagateNode error: ") << "unrecognized node operation type "
                  << strings::info("`" + node->operation_type_ + "`") << std::endl;
        exit(-1);
    }
}

// NOTE: this is a BINARY operation
//       might change in the future?
void multiplyGradient(std::shared_ptr<Node> node) {
    const std::string& name = node->arg_order_[0];
    std::shared_ptr<Node> child = node->children_[node->arg_order_[0]];

    const std::string& other_name = node->arg_order_[1];
    std::shared_ptr<Node> other_child = node->children_[node->arg_order_[1]];

    // d{node} / d{children}
    buffer_ops::multiply(node->gradient_, child->gradient_, child->gradient_);
    buffer_ops::multiply(node->gradient_, other_child->gradient_, other_child->gradient_);

    // d{children}
    //
    // special case for multiply(x, x)
    if (name == other_name) {
        buffer_ops::multiplyConstant(child->output_, 2., child->gradient_);
    } else {
        buffer_ops::multiply(other_child->output_, child->gradient_, child->gradient_);
        buffer_ops::multiply(child->output_, other_child->gradient_, other_child->gradient_);
    }
}

void inputGradient(std::shared_ptr<Node> node) {
}

void tensorGradient(std::shared_ptr<Node> node) {
}

void addGradient(std::shared_ptr<Node> node) {
}

void subtractGradient(std::shared_ptr<Node> node) {
}

void divideGradient(std::shared_ptr<Node> node) {
}

void sqrtGradient(std::shared_ptr<Node> node) {
}

void expGradient(std::shared_ptr<Node> node) {
}

void powGradient(std::shared_ptr<Node> node) {
}

void matmulGradient(std::shared_ptr<Node> node) {
}

void constantGradient(std::shared_ptr<Node> node) {
}

void normalGradient(std::shared_ptr<Node> node) {
}

void onesGradient(std::shared_ptr<Node> node) {
}

void sigmoidGradient(std::shared_ptr<Node> node) {
}

void reluGradient(std::shared_ptr<Node> node) {
}

}  // namespace gradient
