#include "kernel.h"

#include <iostream>
#include <memory>

#include "buffer.h"
#include "graph.h"
#include "ops.h"

namespace kernel {

// TODO: how will broadcasting be handled?
// TODO: figure something out with the float templates
// TODO: parallelization
// TODO: gpu programming

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

// naive implementation
// TODO: make it not naive
void matmul(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> left_node = node->children_[node->arg_order_[0]];
    std::shared_ptr<Node> right_node = node->children_[node->arg_order_[1]];

    int l = left_node->shape_.size();
    int r = right_node->shape_.size();

    std::vector<int> l_index(l, 0);
    std::vector<int> r_index(r, 0);
    std::vector<int> out_index(l, 0);

    // TODO: shapes beyond 2-D

    for (int i = 0; i < left_node->shape_[l - 2]; i++) {
        for (int j = 0; j < right_node->shape_[r - 1]; j++) {
            float dot = 0;
            for (int k = 0; k < right_node->shape_[r - 2]; k++) {
                l_index[l - 2] = i;
                l_index[l - 1] = k;

                r_index[r - 2] = k;
                r_index[r - 1] = j;

                float a = left_node->output_->getIndex<float>(calculateIndex(l_index, left_node->shape_));
                float b = right_node->output_->getIndex<float>(calculateIndex(r_index, right_node->shape_));

                dot += a * b;
            }

            out_index[l - 2] = i;
            out_index[l - 1] = j;

            node->output_->setIndex(calculateIndex(out_index, node->shape_), (void*)(&dot));
        }
    }
}

void constant(std::shared_ptr<Node> node) {
    // pretty sure nothing needs done here
}

void tensor(std::shared_ptr<Node> node) {}

void normal(std::shared_ptr<Node> node) {}

void sigmoid(std::shared_ptr<Node> node) {
    size_t size = 1;
    for (int i : node->shape_) {
        size *= i;
    }

    std::shared_ptr<Node> input = node->children_[node->arg_order_[0]];
    // probably bad approximation per ChatGPT lol
    // TODO: change this, obviously
    for (size_t i = 0; i < size; i++) {
        float x = input->output_->getIndex<float>(i);
        float output = 0;
        if (x < -4) {
            output = 0;
        } else if (x > 4) {
            output = 1;
        } else {
            float x2 = x * x;
            output = x * (0.5 + 0.15012 * x2) / (1 + 0.20162 * x2);
        }

        node->output_->setIndex(i, (void*)(&output));
    }
}

void relu(std::shared_ptr<Node> node) {
    size_t size = 1;
    for (int i : node->shape_) {
        size *= i;
    }

    std::shared_ptr<Node> input = node->children_[node->arg_order_[0]];
    for (size_t i = 0; i < size; i++) {
        float x = input->output_->getIndex<float>(i);
        float output = x > 0 ? x : 0;

        node->output_->setIndex(i, (void*)(&output));
    }
}

}  // namespace kernel
