#include "grad.h"

#include <memory>

#include "broadcasting.h"
#include "buffer.h"
#include "buffer_ops.h"
#include "graph.h"
#include "kernel.h"
#include "ops.h"
#include "string_utils.h"

// TODO: implement the unimplemented
// TODO: these can probably be heavily optimized
// TODO: what do we do with the constant functions? e.g. normal, tensor, etc.
// TODO: broadcasting needs better accounted for in grad calculations
//
// TODO: we need better broadcasting checks and guarantees
//
// NOTE: in all of these, they're assumed to have the structure of
//       f(a, b, ...) = ...
//       where f is the node `node` passed to the function as the argument
//       a, b, ... etc. are the given inputs to `node` in their respective order

// MAJOR TODO: gradients should NOT be initialized as 1 with their values mutliplied in!
//             They should be set at their first calculation, then multiplied with the backpropagated gradient!
//             If a function is not doing that, then it is incorrectly calculating the gradient!

namespace gradient {

void _propagate_current_grad(std::shared_ptr<Node> node, std::shared_ptr<Node> child) {
    if (child->operation_type_ != operations::constant) {
        // accumulation for broadcasted operations
        std::shared_ptr<GraphBuffer> lesser = child->gradient_;
        std::shared_ptr<GraphBuffer> greater = node->gradient_;

        auto [node_shape, child_shape] = broadcasting::padVectors(node->gradient_->shape(), child->gradient_->shape());

        for (int i = 0; i < node_shape.size(); i++) {
            if (node_shape[i] < child_shape[i]) {
                lesser = node->gradient_;
                greater = child->gradient_;
            } else if (node_shape[i] > child_shape[i]) {
                lesser = child->gradient_;
                greater = node->gradient_;
            }
        }

        // we only worry about the case in which `node` > `child`
        // which means the gradient will need to be reduced to the size of `child`
        //
        // otherwise, `node` will be broadcasted up to `child` which is already handled by the `buffer_ops`
        if (lesser == child->gradient_ && greater == node->gradient_) {
            std::vector<int> reduction_indices;
            for (int i = 0; i < child_shape.size(); i++) {
                if (child_shape[i] == 1) {
                    reduction_indices.push_back(i);
                }
            }

            std::shared_ptr<GraphBuffer> reduced_grad(new GraphBuffer(child_shape, DTYPE::float32));

            buffer_ops::reduceSum(node->gradient_, reduced_grad, reduction_indices);
            buffer_ops::multiply(reduced_grad, child->gradient_, child->gradient_);
        } else {
            buffer_ops::multiply(node->gradient_, child->gradient_, child->gradient_);
        }
    }
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

    // d{children}
    //
    // special case for multiply(x, x)
    if (name == other_name) {
        buffer_ops::multiply(child->output_, 2., child->gradient_);
    } else {
        buffer_ops::multiplyAndReduce(other_child->output_, child->gradient_, child->gradient_);
        buffer_ops::multiplyAndReduce(child->output_, other_child->gradient_, other_child->gradient_);
    }

    // d{node} / d{children}
    _propagate_current_grad(node, child);
    _propagate_current_grad(node, other_child);
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

    // for f(a, b) = a / b
    // df/da = 1 / b
    // df/db = -ab^-2

    // df/da
    buffer_ops::reciprocal(denominator->output_, numerator->gradient_);

    // df/db
    buffer_ops::pow(denominator->output_, 2., denominator->gradient_);
    buffer_ops::multiplyAndReduce(denominator->gradient_, numerator->output_, denominator->gradient_);
    buffer_ops::multiply(denominator->gradient_, -1., denominator->gradient_);

    _propagate_current_grad(node, numerator);
    _propagate_current_grad(node, denominator);
}

void sqrtGradient(std::shared_ptr<Node> node) {
    // f(a) = a ^ (1 / 2)
    // df/da = 1 / (2 * a ^ (1 / 2))

    std::shared_ptr<Node> arg = node->children_[node->arg_order_[0]];

    buffer_ops::multiply(arg->output_, 2., arg->gradient_);
    buffer_ops::reciprocal(arg->gradient_, arg->gradient_);

    _propagate_current_grad(node, arg);
}

void expGradient(std::shared_ptr<Node> node) {
    // f(a) = e ^ a
    // df/da = e ^ a

    std::shared_ptr<Node> arg = node->children_[node->arg_order_[0]];

    buffer_ops::multiplyAndReduce(arg->gradient_, arg->output_, arg->gradient_);

    _propagate_current_grad(node, arg);
}

void powGradient(std::shared_ptr<Node> node) {
    // f(a, b) = a ^ b
    // df/da = ba ^ (b - 1)
    // df/db = ln(a) * a ^ b

    std::shared_ptr<Node> base = node->children_[node->arg_order_[0]];
    std::shared_ptr<Node> power = node->children_[node->arg_order_[1]];

    // df/da
    std::shared_ptr<GraphBuffer> minus_one(new GraphBuffer(power->output_->shape(), power->output_->dtype()));
    std::shared_ptr<GraphBuffer> base_grad_staging(new GraphBuffer(base->gradient_->shape(), DTYPE::float32));

    buffer_ops::add(power->output_, -1, minus_one);
    buffer_ops::pow(base->output_, minus_one, base_grad_staging);
    buffer_ops::multiply(base_grad_staging, power->output_, base_grad_staging);
    buffer_ops::copy(base_grad_staging, base->gradient_);

    // df/db
    std::shared_ptr<GraphBuffer> power_grad_staging(new GraphBuffer(power->gradient_->shape(), DTYPE::float32));
    if (power->gradient_->size() == 1) {
        std::shared_ptr<GraphBuffer> temp(new GraphBuffer(base->output_->shape(), base->output_->dtype()));
        buffer_ops::ln(base->output_, temp);
        float grad_sum = buffer_ops::reduceSum(temp);
        power->gradient_->setIndex(0, (void*)(&grad_sum));
    } else {
        buffer_ops::ln(base->output_, power_grad_staging);
        buffer_ops::multiplyAndReduce(power_grad_staging, node->output_, power_grad_staging);
        buffer_ops::copy(power_grad_staging, power->gradient_);
    }

    _propagate_current_grad(node, base);
    _propagate_current_grad(node, power);
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

    auto [a_transpose_shape, b_transpose_shape] = broadcasting::padVectors(a->output_->shape(), b->output_->shape());

    std::swap(a_transpose_shape[a_transpose_shape.size() - 2], a_transpose_shape[a_transpose_shape.size() - 1]);
    std::swap(b_transpose_shape[b_transpose_shape.size() - 2], b_transpose_shape[b_transpose_shape.size() - 1]);

    std::shared_ptr<GraphBuffer> a_transpose(new GraphBuffer(a_transpose_shape, a->output_->dtype()));
    std::shared_ptr<GraphBuffer> b_transpose(new GraphBuffer(b_transpose_shape, b->output_->dtype()));

    std::vector<int> perm(a->gradient_->shape().size());
    for (int i = 0; i < a->gradient_->shape().size(); i++) {
        perm[i] = i;
    }

    std::swap(perm[perm.size() - 2], perm[perm.size() - 1]);

    std::shared_ptr<GraphBuffer> a_staging_grad(new GraphBuffer(a->gradient_->shape(), DTYPE::float32));
    std::shared_ptr<GraphBuffer> b_staging_grad(new GraphBuffer(b->gradient_->shape(), DTYPE::float32));

    // df/dA
    buffer_ops::transpose(b->output_, b_transpose, perm);
    buffer_ops::matmul(node->gradient_, b_transpose, a_staging_grad, node->shape_, b_transpose->shape_, a->shape_);

    buffer_ops::multiply(a_staging_grad, a->gradient_, a->gradient_);

    // df/dB
    buffer_ops::transpose(a->output_, a_transpose, perm);
    if (b->gradient_->shape().size() == a_transpose->shape().size()) {
        buffer_ops::matmul(a_transpose, node->gradient_, b_staging_grad, a_transpose->shape_, node->shape_, b->shape_);
        buffer_ops::multiply(b_staging_grad, b->gradient_, b->gradient_);
    }
    // happened in the forward pass, for e.g. batches
    else {
        const std::vector<int> gradient_shape = node->gradient_->shape();
        const std::vector<int>& transpose_shape = a_transpose->shape();

        std::vector<int> transpose_matmul_shape;
        for (int i = 0; i < gradient_shape.size() - 2; i++) {
            transpose_matmul_shape.push_back(gradient_shape[i]);
        }

        transpose_matmul_shape.push_back(transpose_shape[transpose_shape.size() - 2]);
        transpose_matmul_shape.push_back(gradient_shape.back());

        std::vector<int> reduced;
        for (int i = 1; i < transpose_matmul_shape.size(); i++) {
            reduced.push_back(transpose_matmul_shape[i]);
        }

        std::shared_ptr<GraphBuffer> transpose_matmul(new GraphBuffer(transpose_matmul_shape, DTYPE::float32));

        buffer_ops::matmul(a_transpose, node->gradient_, transpose_matmul, a_transpose->shape_, node->shape_,
                           transpose_matmul_shape);

        const std::vector<int> b_gradient_shape = b->gradient_->shape();
        // this is gross and I'd really rather we not do this
        std::vector<int> temp_shape = broadcasting::padVector(b_gradient_shape, transpose_matmul_shape.size());
        b_staging_grad->shape_ = temp_shape;

        // why is the reduction index array just `{0}` ???
        buffer_ops::reduceSum(transpose_matmul, b_staging_grad, {0});
        buffer_ops::multiply(b_staging_grad, b->gradient_, b->gradient_);
    }
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
            float output = _a->getIndex<float>(index) > EPSILON ? 1 : 0;
            _out->setIndex(index, (void*)(&output));
        },
        a->output_, a->gradient_);

    _propagate_current_grad(node, a);
}

void reduce_sumGradient(std::shared_ptr<Node> node) {
    _propagate_current_grad(node, node->children_[node->arg_order_[0]]);
}

void conv2dGradient(std::shared_ptr<Node> node) {
}

}  // namespace gradient
