#include "grad.h"

#include <memory>

#include "buffer.h"
#include "buffer_ops.h"
#include "graph.h"
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
    // f(A, B) = A @ B
    // df/dA = all elements of each column j are the sum of the elements in row j of B
    // df/dB = all elements of each row i are the sum of the elements in column i of A

    std::shared_ptr<Node> a = node->children_[node->arg_order_[0]];
    std::shared_ptr<Node> b = node->children_[node->arg_order_[1]];

    const std::vector<int>& shape_a = a->shape_;
    const std::vector<int>& shape_b = b->shape_;

    size_t matrix_count = 1;
    for (int i = 0; i < shape_a.size() - 2; i++) {
        matrix_count *= shape_a[i];

        if (shape_a[i] != shape_b[i]) {
            std::cerr << strings::error("gradient::matmaulGradient error: ")
                      << "incompatible input/output batch shapes, got " << strings::info(strings::vecToString(shape_a))
                      << " and " << strings::info(strings::vecToString(shape_b)) << std::endl;
            exit(-1);
        }
    }

    int a_row_count = shape_a[shape_a.size() - 2];
    int a_col_count = shape_a[shape_a.size() - 1];

    int b_row_count = shape_b[shape_b.size() - 2];
    int b_col_count = shape_b[shape_b.size() - 1];

    size_t a_matrix_size = a_row_count * a_col_count;
    size_t b_matrix_size = b_row_count * b_col_count;

    // df/dA

    // for each batch
    for (size_t i = 0; i < matrix_count; i++) {
        // for each column of a
        for (size_t a_c = 0; a_c < a_col_count; a_c++) {
            float sum = 0;
            // sum row a_c of b
            for (int b_c = 0; b_c < b_col_count; b_c++) {
                sum += b->output_->getIndex<float>(i * b_matrix_size + (a_c * b_col_count + b_c));
            }

            // assign the column
            for (int a_r = 0; a_r < a_row_count; a_r++) {
                a->gradient_->setIndex(i * a_matrix_size + (a_r * a_col_count + a_c), (void*)(&sum));
            }
        }
    }

    // df/dB
    for (size_t i = 0; i < matrix_count; i++) {
        // for each row of b
        for (int b_r = 0; b_r < b_row_count; b_r++) {
            float sum = 0;
            // sum column b_r of a
            for (int a_r = 0; a_r < a_row_count; a_r++) {
                sum += a->output_->getIndex<float>(i * a_matrix_size + (a_r * a_row_count + b_r));
            }

            // assign the row
            for (int b_c = 0; b_c < b_col_count; b_c++) {
                b->gradient_->setIndex(i * b_matrix_size + (b_r * b_col_count + b_c), (void*)(&sum));
            }
        }
    }
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
