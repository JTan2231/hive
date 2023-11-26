#include "grad.h"

#include <memory>

#include "buffer.h"
#include "buffer_ops.h"
#include "graph.h"
#include "kernel.h"
#include "ops.h"
#include "string_utils.h"

// TODO: implement the unimplemented
// TODO: these can probably be heavily optimized
// TODO: what do we do with the constant functions? e.g. normal, tensor, etc.
//
// NOTE: in all of these, they're assumed to have the structure of
//       f(a, b, ...) = ...
//       where f is the node `node` passed to the function as the argument
//       a, b, ... etc. are the given inputs to `node` in their respective order

namespace gradient {

void _propagate_current_grad(std::shared_ptr<Node> node, std::shared_ptr<Node> child) {
    buffer_ops::multiply(node->gradient_, child->gradient_, child->gradient_);
}

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
    _propagate_current_grad(node, child);
    _propagate_current_grad(node, other_child);

    // d{children}
    //
    // special case for multiply(x, x)
    if (name == other_name) {
        buffer_ops::multiply(child->output_, 2., child->gradient_);
    } else {
        buffer_ops::multiply(other_child->output_, child->gradient_, child->gradient_);
        buffer_ops::multiply(child->output_, other_child->gradient_, other_child->gradient_);
    }
}

void inputGradient(std::shared_ptr<Node> node) {
}

void tensorGradient(std::shared_ptr<Node> node) {
}

// this is just 1, no action needed
void addGradient(std::shared_ptr<Node> node) {
    for (auto& [name, child] : node->children_) {
        _propagate_current_grad(node, child);
    }
}

void subtractGradient(std::shared_ptr<Node> node) {
    for (auto& [name, child] : node->children_) {
        _propagate_current_grad(node, child);
    }

    // only the first remains positive
    for (int i = 1; i < node->arg_order_.size(); i++) {
        std::shared_ptr<Node> child = node->children_[node->arg_order_[i]];
        buffer_ops::multiply(child->gradient_, -1., child->gradient_);
    }
}

void divideGradient(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> numerator = node->children_[node->arg_order_[0]];
    std::shared_ptr<Node> denominator = node->children_[node->arg_order_[1]];

    _propagate_current_grad(node, numerator);
    _propagate_current_grad(node, denominator);

    // for f(a, b) = a / b
    // df/da = 1 / b
    // df/db = -ab^-2

    // df/da
    buffer_ops::reciprocal(denominator->output_, numerator->gradient_);

    // df/db
    buffer_ops::pow(denominator->output_, 2., denominator->gradient_);
    buffer_ops::multiply(denominator->gradient_, numerator->output_, denominator->gradient_);
    buffer_ops::multiply(denominator->gradient_, -2., denominator->gradient_);
}

void sqrtGradient(std::shared_ptr<Node> node) {
    // f(a) = a ^ (1 / 2)
    // df/da = 1 / (2 * a ^ (1 / 2))

    std::shared_ptr<Node> arg = node->children_[node->arg_order_[0]];

    _propagate_current_grad(node, arg);

    buffer_ops::multiply(arg->output_, 2., arg->gradient_);
    buffer_ops::reciprocal(arg->gradient_, arg->gradient_);
}

void expGradient(std::shared_ptr<Node> node) {
    // f(a) = e ^ a
    // df/da = e ^ a

    std::shared_ptr<Node> arg = node->children_[node->arg_order_[0]];

    _propagate_current_grad(node, arg);

    buffer_ops::multiply(arg->gradient_, arg->output_, arg->gradient_);
}

void powGradient(std::shared_ptr<Node> node) {
    // f(a, b) = a ^ b
    // df/da = ba ^ (b - 1)
    // df/db = ln(a) * a ^ b

    std::shared_ptr<Node> base = node->children_[node->arg_order_[0]];
    std::shared_ptr<Node> power = node->children_[node->arg_order_[1]];

    _propagate_current_grad(node, base);
    _propagate_current_grad(node, power);

    // df/da
    // this is disgusting
    buffer_ops::pow(base->output_, power->output_, base->gradient_);
    buffer_ops::divide(base->gradient_, base->output_, base->gradient_);  // to get the (b - 1) in the exponent
    buffer_ops::multiply(base->gradient_, power->output_, base->gradient_);

    // df/db
    buffer_ops::ln(base->output_, power->gradient_);
    buffer_ops::multiply(power->gradient_, node->output_, power->gradient_);
}

// TODO: broadcasting
// TODO: this can definitely be optimized
// TODO: lotta repeated code here...
void matmulGradient(std::shared_ptr<Node> node) {
    // basing this off https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_grad.py#L1694
    //
    // A = [n x m]
    // B = [m x p]
    // f(A, B) = A @ B [n x p]
    // grad = [n x p]
    // df/dA = grad @ B ^ T
    // df/dB = A ^ T @ grad

    std::shared_ptr<Node> a = node->children_[node->arg_order_[0]];
    std::shared_ptr<Node> b = node->children_[node->arg_order_[1]];

    std::shared_ptr<Buffer> a_transpose(new Buffer(a->output_->size(), a->output_->dtype()));
    std::vector<int> a_transpose_shape = a->shape_;
    std::swap(a_transpose_shape[a->shape_.size() - 2], a_transpose_shape[a->shape_.size() - 1]);

    std::shared_ptr<Buffer> b_transpose(new Buffer(b->output_->size(), b->output_->dtype()));
    std::vector<int> b_transpose_shape = b->shape_;
    std::swap(b_transpose_shape[b->shape_.size() - 2], b_transpose_shape[b->shape_.size() - 1]);

    // df/dA
    buffer_ops::transpose(b->output_, b->shape_, b_transpose, b_transpose_shape);
    buffer_ops::matmul(node->gradient_, b_transpose, a->gradient_, node->shape_, b_transpose_shape, a->shape_);

    // df/dB
    buffer_ops::transpose(a->output_, a->shape_, a_transpose, a_transpose_shape);
    buffer_ops::matmul(a_transpose, node->gradient_, b->gradient_, a_transpose_shape, node->shape_, b->shape_);
}

void constantGradient(std::shared_ptr<Node> node) {
}

void normalGradient(std::shared_ptr<Node> node) {
}

void onesGradient(std::shared_ptr<Node> node) {
}

void sigmoidGradient(std::shared_ptr<Node> node) {
    // f(a) = 1 / (1 + exp(-a))
    // df/da = f(a) * (1 - f(a))

    std::shared_ptr<Node> a = node->children_[node->arg_order_[0]];

    buffer_ops::add(node->output_, -1., a->gradient_);
    buffer_ops::multiply(node->output_, a->gradient_, a->gradient_);
}

void reluGradient(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> a = node->children_[node->arg_order_[0]];
    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _out, size_t index) {
            float output = _a->getIndex<float>(index) ? 1 : 0;
            _out->setIndex(index, (void*)(&output));
        },
        node->output_, a->gradient_);
}

}  // namespace gradient
