#include "kernel.h"

#include <iostream>
#include <memory>

#include "broadcasting.h"
#include "buffer.h"
#include "buffer_ops.h"
#include "graph.h"
#include "iterators.h"
#include "ops.h"
#include "string_utils.h"

namespace kernel {

// TODO: how will broadcasting be handled?
// TODO: figure something out with the float templates
// TODO: parallelization
// TODO: gpu programming
// TODO: is there anything we can do to manage precision? is that even an issue?

bool broadcastable(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b) {
    auto as = a->shape();
    auto bs = b->shape();

    if (as.size() > bs.size()) {
        std::vector<int> ones(as.size() - bs.size(), 1);
        bs.insert(bs.begin(), ones.begin(), ones.end());
    } else if (bs.size() > as.size()) {
        std::vector<int> ones(bs.size() - as.size(), 1);
        as.insert(as.begin(), ones.begin(), ones.end());
    }

    for (int i = 0; i < as.size(); i++) {
        if (as[i] != bs[i]) {
            if (as[i] != 1 && bs[i] != 1) {
                return false;
            }
        }
    }

    return true;
}

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

void _element_wise(std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t,
                                      size_t, size_t)>
                       element_function,
                   std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    int padding_size = std::max(a->shape().size(), b->shape().size());
    std::vector<int> a_shape = broadcasting::padVector(a->shape(), padding_size);
    std::vector<int> b_shape = broadcasting::padVector(b->shape(), padding_size);

    bool ab_order = iterators::lesserGreater(a_shape, b_shape);
    const std::vector<int>& lesser = ab_order ? a_shape : b_shape;
    const std::vector<int>& greater = ab_order ? b_shape : a_shape;

    iterators::BroadcastIterator it(lesser, greater);
    while (!it.end()) {
        auto [lesser_index, greater_index] = it.getIndices();
        if (ab_order) {
            element_function(a, b, out, lesser_index, greater_index, greater_index);
        } else {
            element_function(a, b, out, greater_index, lesser_index, greater_index);
        }

        it.increment();
    }
}

void _element_wise(
    std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t, size_t)> element_function,
    std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    int padding_size = std::max(a->shape().size(), out->shape().size());
    std::vector<int> a_shape = broadcasting::padVector(a->shape(), padding_size);
    std::vector<int> b_shape = broadcasting::padVector(out->shape(), padding_size);

    bool ab_order = iterators::lesserGreater(a_shape, b_shape);
    const std::vector<int>& lesser = ab_order ? a_shape : b_shape;
    const std::vector<int>& greater = ab_order ? b_shape : a_shape;

    iterators::BroadcastIterator it(lesser, greater);
    while (!it.end()) {
        auto [lesser_index, greater_index] = it.getIndices();
        if (ab_order) {
            element_function(a, out, lesser_index, greater_index);
        } else {
            element_function(a, out, greater_index, lesser_index);
        }

        it.increment();
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
void matmul(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> left_node = node->children_[node->arg_order_[0]];
    std::shared_ptr<Node> right_node = node->children_[node->arg_order_[1]];

    std::vector<int> l = left_node->shape_;
    std::vector<int> r = right_node->shape_;
    std::vector<int> o = node->shape_;

    buffer_ops::matmul(left_node->output_, right_node->output_, node->output_, l, r, o);
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
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t a_index,
           size_t b_index, size_t out_index) {
            float output = a->getIndex<float>(a_index) + b->getIndex<float>(b_index);
            out->setIndex(out_index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void subtract(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t a_index,
           size_t b_index, size_t out_index) {
            float output = a->getIndex<float>(a_index) - b->getIndex<float>(b_index);
            out->setIndex(out_index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void multiply(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t a_index,
           size_t b_index, size_t out_index) {
            float output = a->getIndex<float>(a_index) * b->getIndex<float>(b_index);
            out->setIndex(out_index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void divide(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t a_index,
           size_t b_index, size_t out_index) {
            if (std::fabs(b->getIndex<float>(b_index)) < EPSILON) {
                std::cerr << strings::error("kernel::divide error: ") << "divide by zero error." << std::endl;
                exit(-1);
            }

            float output = a->getIndex<float>(a_index) / b->getIndex<float>(b_index);
            out->setIndex(out_index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void sqrt(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t in_index, size_t out_index) {
            if (a->getIndex<float>(in_index) < 0) {
                std::cerr << strings::error("kernel::sqrt error: ")
                          << "argument is less than zero. We don't support complex numbers yet!" << std::endl;
                exit(-1);
            }

            float output = std::sqrt(a->getIndex<float>(in_index));
            out->setIndex(out_index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

void exp(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t in_index, size_t out_index) {
            float output = std::exp(a->getIndex<float>(in_index));
            out->setIndex(out_index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

void pow(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t a_index,
           size_t b_index, size_t out_index) {
            float output = std::pow(a->getIndex<float>(a_index), b->getIndex<float>(b_index));
            out->setIndex(out_index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void sigmoid(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t in_index, size_t out_index) {
            float output = 1 / (1 + std::exp(-(a->getIndex<float>(in_index))));
            out->setIndex(out_index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

void relu(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t in_index, size_t out_index) {
            float output = a->getIndex<float>(in_index);
            output = output > 0 ? output : 0;
            out->setIndex(out_index, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

void reduce_sum(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t in_index, size_t out_index) {
            float output = a->getIndex<float>(in_index) + out->getIndex<float>(0);
            out->setIndex(0, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

}  // namespace kernel
