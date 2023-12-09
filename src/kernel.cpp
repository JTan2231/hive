#include "kernel.h"

#include <iostream>
#include <memory>

#include "buffer.h"
#include "graph.h"
#include "ops.h"
#include "string_utils.h"

namespace kernel {

// TODO: how will broadcasting be handled?
// TODO: figure something out with the float templates
// TODO: parallelization
// TODO: gpu programming
// TODO: is there anything we can do to manage precision? is that even an issue?

void computeNode(std::shared_ptr<Node> node) {
    const auto& operationMap = OperationRegistry::GetOperationMap();

    auto it = operationMap.find(node->operation_type_);
    if (it != operationMap.end()) {
        it->second(node);
    } else {
        std::cerr << "kernel::computeNode error: unrecognized node operation type " << node->operation_type_
                  << std::endl;
        exit(-1);
    }
}

void _element_wise(
    std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t)>
        element_function,
    std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    for (size_t i = 0; i < out->size(); i++) {
        element_function(a, b, out, i);
    }
}

void _element_wise(std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t)> element_function,
                   std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    for (size_t i = 0; i < out->size(); i++) {
        element_function(a, out, i);
    }
}

void _element_wise(
    std::function<void(std::shared_ptr<Buffer>, float, std::shared_ptr<Buffer>, size_t)> element_function,
    std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    for (size_t i = 0; i < out->size(); i++) {
        element_function(a, b, out, i);
    }
}

// naive implementation
// TODO: make it not naive
// TODO: broadcasting
void matmul(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> left_node = node->children_[node->arg_order_[0]];
    std::shared_ptr<Node> right_node = node->children_[node->arg_order_[1]];

    int l = left_node->shape_.size();
    int r = right_node->shape_.size();
    int o = node->shape_.size();

    size_t l_matrix_size = left_node->shape_[l - 2] * left_node->shape_[l - 1];
    size_t r_matrix_size = right_node->shape_[r - 2] * right_node->shape_[r - 1];
    size_t o_matrix_size = node->shape_[o - 2] * node->shape_[o - 1];

    // for batched matrix multiplication, only the last two dimensions are considered in the multipliciation
    // the rest of the dimenions just served to act as groupings of matrices
    //
    // NOTE: shape verification if performed in `allocation.cpp`
    size_t matrix_count = 1;
    for (int i = 0; i < left_node->shape_.size() - 2; i++) {
        matrix_count *= left_node->shape_[i];
    }

    for (int m = 0; m < matrix_count; m++) {
        for (int i = 0; i < left_node->shape_[l - 2]; i++) {
            for (int j = 0; j < right_node->shape_[r - 1]; j++) {
                float dot = 0;
                for (int k = 0; k < right_node->shape_[r - 2]; k++) {
                    float a =
                        left_node->output_->getIndex<float>((i * l_matrix_size) + (i * left_node->shape_[l - 1]) + k);
                    float b =
                        right_node->output_->getIndex<float>((i * r_matrix_size) + (k * right_node->shape_[r - 1]) + j);

                    dot += a * b;
                }

                node->output_->setIndex((i * o_matrix_size) + (i * node->shape_[o - 2]) + j, (void*)(&dot));
            }
        }
    }
}

void input(std::shared_ptr<Node> node) {
}

void constant(std::shared_ptr<Node> node) {
}

void tensor(std::shared_ptr<Node> node) {
}

void normal(std::shared_ptr<Node> node) {
}

void ones(std::shared_ptr<Node> node) {
}

void add(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t index) {
            float output = a->getIndex<float>(index) + b->getIndex<float>(index);
            out->setIndex(index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void subtract(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t index) {
            float output = a->getIndex<float>(index) - b->getIndex<float>(index);
            out->setIndex(index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void multiply(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t index) {
            float output = a->getIndex<float>(index) * b->getIndex<float>(index);
            out->setIndex(index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void divide(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t index) {
            if (std::fabs(b->getIndex<float>(index)) < EPSILON) {
                std::cerr << strings::error("kernel::divide error: ") << "divide by zero error." << std::endl;
                exit(-1);
            }

            float output = a->getIndex<float>(index) / b->getIndex<float>(index);
            out->setIndex(index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void sqrt(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t index) {
            if (a->getIndex<float>(index) < 0) {
                std::cerr << strings::error("kernel::sqrt error: ")
                          << "argument is less than zero. We don't support complex numbers yet!" << std::endl;
                exit(-1);
            }

            float output = std::sqrt(a->getIndex<float>(index));
            out->setIndex(index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

void exp(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t index) {
            float output = std::exp(a->getIndex<float>(index));
            out->setIndex(index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

void pow(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t index) {
            float output = std::pow(a->getIndex<float>(index), b->getIndex<float>(index));
            out->setIndex(index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void sigmoid(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t index) {
            float output = 1 / (1 + std::exp(-(a->getIndex<float>(index))));
            out->setIndex(index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

void relu(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t index) {
            float output = a->getIndex<float>(index);
            output = output > 0 ? output : 0;
            out->setIndex(index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

}  // namespace kernel
