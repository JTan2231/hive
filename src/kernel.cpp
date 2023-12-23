#include "kernel.h"

#include <iostream>
#include <memory>

#include "broadcasting.h"
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

BroadcastIterator::BroadcastIterator(std::vector<int> lesser, std::vector<int> greater) {
    if (lesser.size() != greater.size()) {
        std::cerr << strings::error("BroadcastIterator::BroadcastIterator error: ") << "shape sizes must be equal, got "
                  << strings::info(strings::vecToString(lesser)) << " and "
                  << strings::info(strings::vecToString(greater)) << std::endl;
        exit(-1);
    }

    for (int i = 0; i < lesser.size(); i++) {
        if (lesser[i] != greater[i] && lesser[i] != 1 && greater[i] != 1) {
            std::cerr << strings::error("BroadcastIterator::BroadcastIterator error: ")
                      << "dimensions must be equal or 1, got " << strings::info(strings::vecToString(lesser)) << " and "
                      << strings::info(strings::vecToString(greater)) << std::endl;
            exit(-1);
        }
    }

    lesser_ = lesser;
    greater_ = greater;

    lesser_current_ = std::vector<int>(lesser.size(), 0);
    greater_current_ = std::vector<int>(greater.size(), 0);
}

bool BroadcastIterator::end() {
    return end_;
}

void BroadcastIterator::print() {
    std::cout << "- greater: " << strings::info(strings::vecToString(greater_current_)) << std::endl;
    std::cout << "- lesser: " << strings::info(strings::vecToString(lesser_current_)) << std::endl;
}

// returns {lesser_index, greater_index}
std::pair<size_t, size_t> BroadcastIterator::getIndices() {
    return {getIndex(false), getIndex(true)};
}

size_t BroadcastIterator::getIndex(bool greater) {
    const std::vector<int>& indices = greater ? greater_current_ : lesser_current_;
    const std::vector<int>& shape = greater ? greater_ : lesser_;

    size_t index = indices.back();
    size_t prefix = shape.back();
    for (int i = shape.size() - 2; i > -1; i--) {
        index += indices[i] * prefix;
        prefix *= shape[i];
    }

    return index;
}

void BroadcastIterator::propagateChanges() {
    std::vector<int> change_indices(greater_.size(), 0);
    int change = 0;
    for (int i = greater_current_.size() - 1; i > -1; i--) {
        greater_current_[i] += change;
        change_indices[i] += change;

        if (greater_current_[i] == greater_[i]) {
            change = 1;
        } else if (greater_current_[i] == -1) {
            change = -1;
        } else {
            change = 0;
        }
    }

    for (int i = lesser_current_.size() - 1; i > -1; i--) {
        if (lesser_[i] != 1) {
            lesser_current_[i] += change_indices[i];
        } else if (lesser_current_[i] == 1) {
            lesser_current_[i] = 0;
        }
    }
}

void BroadcastIterator::resetOutOfBounds() {
    for (int i = 0; i < greater_.size(); i++) {
        if (greater_current_[i] == greater_[i]) {
            greater_current_[i] = 0;
        } else if (greater_current_[i] == -1) {
            greater_current_[i] = greater_[i] - 1;
        }

        if (lesser_current_[i] == lesser_[i]) {
            lesser_current_[i] = 0;
        } else if (lesser_current_[i] == -1) {
            lesser_current_[i] = lesser_[i] - 1;
        }
    }
}

void BroadcastIterator::updateEnd() {
    end_ = true;
    for (int i = 0; i < greater_.size(); i++) {
        if (greater_current_[i] != greater_[i] - 1) {
            end_ = false;
            return;
        }
    }
}

bool BroadcastIterator::increment() {
    int n = greater_.size();

    greater_current_[n - 1]++;
    lesser_current_[n - 1]++;
    if (greater_current_[n - 1] == greater_[n - 1]) {
        propagateChanges();
    }

    resetOutOfBounds();
    updateEnd();

    return !end_;
}

bool BroadcastIterator::decrement() {
    int n = greater_.size();

    greater_current_[n - 1]--;
    lesser_current_[n - 1]--;
    if (greater_current_[n - 1] == greater_[n - 1]) {
        propagateChanges();
    }

    resetOutOfBounds();
    updateEnd();

    return !end_;
}

bool lesserGreater(std::vector<int> a, std::vector<int> b) {
    for (int i = 0; i < a.size(); i++) {
        if (a[i] < b[i]) {
            return true;
        }
    }

    return false;
}

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
        std::cout << strings::error("COMPUTING NODE ") << strings::info(node->name_) << std::endl;
        it->second(node);
        std::cout << strings::error("FINISHED COMPUTING NODE ") << strings::info(node->name_) << std::endl;
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
    std::cout << strings::debug("ENTER BROADCASTED ELEMENT WISE") << std::endl;

    int padding_size = std::max(a->shape().size(), b->shape().size());
    std::vector<int> a_shape = broadcasting::padVector(a->shape(), padding_size);
    std::vector<int> b_shape = broadcasting::padVector(b->shape(), padding_size);

    bool ab_order = lesserGreater(a_shape, b_shape);
    const std::vector<int>& lesser = ab_order ? a_shape : b_shape;
    const std::vector<int>& greater = ab_order ? b_shape : a_shape;

    std::cout << b->size() << ", " << a->size() << ", " << out->size() << std::endl;
    std::cout << strings::vecToString(lesser) << ", " << strings::vecToString(greater) << std::endl;

    BroadcastIterator it(lesser, greater);
    while (!it.end()) {
        auto [lesser_index, greater_index] = it.getIndices();
        if (ab_order) {
            element_function(a, b, out, lesser_index, greater_index, greater_index);
        } else {
            element_function(a, b, out, greater_index, lesser_index, greater_index);
        }

        it.increment();
    }

    std::cout << strings::debug("EXIT BROADCASTED ELEMENT WISE") << std::endl;
}

void _element_wise(std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t)> element_function,
                   std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    std::cout << strings::debug("ENTER UNARY ELEMENT WISE") << std::endl;
    for (size_t i = 0; i < out->size(); i++) {
        element_function(a, out, i);
    }
    std::cout << strings::debug("EXIT UNARY ELEMENT WISE") << std::endl;
}

void _element_wise(
    std::function<void(std::shared_ptr<Buffer>, float, std::shared_ptr<Buffer>, size_t)> element_function,
    std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    std::cout << strings::debug("ENTER BINARY ELEMENT WISE") << std::endl;
    for (size_t i = 0; i < out->size(); i++) {
        element_function(a, b, out, i);
    }
    std::cout << strings::debug("EXIT BINARY ELEMENT WISE") << std::endl;
}

// naive implementation
// TODO: make it not naive
void matmul(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> left_node = node->children_[node->arg_order_[0]];
    std::shared_ptr<Node> right_node = node->children_[node->arg_order_[1]];

    int l = left_node->shape_.size();
    int r = right_node->shape_.size();
    int o = node->shape_.size();

    size_t l_matrix_size = left_node->shape_[l - 2] * left_node->shape_[l - 1];
    size_t r_matrix_size = right_node->shape_[r - 2] * right_node->shape_[r - 1];
    size_t o_matrix_size = node->shape_[o - 2] * node->shape_[o - 1];

    // for batched matrix multiplication, only the last two dimensions are considered in the multiplication
    // the rest of the dimenions just served to act as groupings of matrices
    //
    // NOTE: shape verification is performed in `allocation.cpp`
    size_t matrix_count = 1;
    for (int i = 0; i < left_node->shape_.size() - 2; i++) {
        matrix_count *= left_node->shape_[i];
    }

    // NOTE: broadcasting isn't really tested all that well
    std::vector<int> l_batch_shape;
    std::vector<int> r_batch_shape;

    for (int i = 0; i < left_node->shape_.size() - 2 || i < right_node->shape_.size() - 2; i++) {
        if (i < left_node->shape_.size() - 2) {
            l_batch_shape.push_back(left_node->shape_[i]);
        }

        if (i < right_node->shape_.size() - 2) {
            r_batch_shape.push_back(right_node->shape_[i]);
        }
    }

    bool l_is_lesser = false;

    if (l_batch_shape.size() < r_batch_shape.size()) {
        std::vector<int> ones(r_batch_shape.size() - l_batch_shape.size(), 1);
        l_batch_shape.insert(l_batch_shape.begin(), ones.begin(), ones.end());
    } else if (r_batch_shape.size() < l_batch_shape.size()) {
        std::vector<int> ones(l_batch_shape.size() - r_batch_shape.size(), 1);
        r_batch_shape.insert(r_batch_shape.begin(), ones.begin(), ones.end());
    }

    for (int i = 0; i < l_batch_shape.size(); i++) {
        if (l_batch_shape[i] < r_batch_shape[i]) {
            l_is_lesser = true;
        } else {
            l_is_lesser = false;
        }
    }

    BroadcastIterator it =
        l_is_lesser ? BroadcastIterator(l_batch_shape, r_batch_shape) : BroadcastIterator(r_batch_shape, l_batch_shape);

    while (!it.end()) {
        auto [lesser_index, greater_index] = it.getIndices();
        size_t left_index = l_is_lesser ? lesser_index : greater_index;
        size_t right_index = l_is_lesser ? greater_index : lesser_index;
        size_t out_index = greater_index;

        for (int i = 0; i < left_node->shape_[l - 2]; i++) {
            for (int j = 0; j < right_node->shape_[r - 1]; j++) {
                float dot = 0;
                for (int k = 0; k < right_node->shape_[r - 2]; k++) {
                    float a = left_node->output_->getIndex<float>((left_index * l_matrix_size) +
                                                                  (i * left_node->shape_[l - 1]) + k);
                    float b = right_node->output_->getIndex<float>((right_index * r_matrix_size) +
                                                                   (k * right_node->shape_[r - 1]) + j);

                    dot += a * b;
                }

                node->output_->setIndex((out_index * o_matrix_size) + (i * node->shape_[o - 2]) + j, (void*)(&dot));
            }
        }

        it.increment();
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
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out, size_t a_index,
           size_t b_index, size_t out_index) {
            float output = std::pow(a->getIndex<float>(a_index), b->getIndex<float>(b_index));
            out->setIndex(out_index, (void*)(&output));
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

void reduce_sum(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t index) {
            float output = a->getIndex<float>(index) + out->getIndex<float>(0);
            out->setIndex(0, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->output_);
}

}  // namespace kernel
