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

// TODO: figure something out with the float templates
// TODO: parallelization
// TODO: gpu programming
// TODO: is there anything we can do to manage precision? is that even an issue?
//
// NOTE: shape validation and creation is largely handled in `allocation.cpp`
//       we _shouldn't_ need to worry about that here

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

void _element_wise(binary_lambda element_function, std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b,
                   std::shared_ptr<Buffer> out) {
    auto [a_broadcasted, b_broadcasted] = broadcasting::makeBroadcastable(a, b);
    std::vector<int> out_shape_broadcasted = broadcasting::broadcastedOutputShape(a_broadcasted, b_broadcasted);
    std::shared_ptr<BroadcastedBuffer> out_broadcasted(new BroadcastedBuffer(out, out_shape_broadcasted));
    iterators::IndexIterator it(out_shape_broadcasted);

    while (!it.end()) {
        element_function(a_broadcasted, b_broadcasted, out_broadcasted, it.getIndices());
        it.increment();
    }
}

void _element_wise(std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t)> element_function,
                   std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    if (a->size() != out->size()) {
        std::cerr << strings::error("kernel::_element_wise error: ")
                  << "unary operation mismatching sizes (can't be broadcasted), got "
                  << strings::info(std::to_string(a->size())) << " and " << strings::info(std::to_string(out->size()))
                  << std::endl;
        exit(-1);
    }

    for (size_t i = 0; i < a->size(); i++) {
        element_function(a, out, i);
    }
}

void _element_wise(
    std::function<void(std::shared_ptr<Buffer>, float, std::shared_ptr<Buffer>, size_t)> element_function,
    std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    if (a->size() != out->size()) {
        std::cerr << strings::error("kernel::_element_wise error: ")
                  << "unary operation mismatching sizes (can't be broadcasted), got "
                  << strings::info(std::to_string(a->size())) << " and " << strings::info(std::to_string(out->size()))
                  << std::endl;
        exit(-1);
    }

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
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out,
           const std::vector<int>& indices) {
            float output = a->getIndex<float>(indices) + b->getIndex<float>(indices);
            out->setIndex(indices, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void subtract(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out,
           const std::vector<int>& indices) {
            float output = a->getIndex<float>(indices) - b->getIndex<float>(indices);
            out->setIndex(indices, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void multiply(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out,
           const std::vector<int>& indices) {
            float output = a->getIndex<float>(indices) * b->getIndex<float>(indices);
            out->setIndex(indices, (void*)(&output));
        },
        node->children_[node->arg_order_[0]]->output_, node->children_[node->arg_order_[1]]->output_, node->output_);
}

void divide(std::shared_ptr<Node> node) {
    _element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out,
           const std::vector<int>& indices) {
            if (std::fabs(b->getIndex<float>(indices)) < EPSILON) {
                std::cerr << strings::error("kernel::divide error: ") << "divide by zero error." << std::endl;
                exit(-1);
            }

            float output = a->getIndex<float>(indices) / b->getIndex<float>(indices);
            out->setIndex(indices, (void*)(&output));
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
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out,
           const std::vector<int>& indices) {
            float output = std::pow(a->getIndex<float>(indices), b->getIndex<float>(indices));
            out->setIndex(indices, (void*)(&output));
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
    std::shared_ptr<GraphBuffer> a = node->children_[node->arg_order_[0]]->output_;
    std::shared_ptr<GraphBuffer> out = node->output_;
    for (size_t i = 0; i < a->size(); i++) {
        float output = a->getIndex<float>(i) + out->getIndex<float>(0);
        out->setIndex(0, (void*)(&output));
    }
}

// `input_image` has shape [batch_dims..., hi, wi, c] where c is the number of channels
// `kernel` has shape [hk, wk, o] where o is the number of output filters
//
// NOTE: this DOES NOT support batched kernels
void conv2d(std::shared_ptr<Node> node) {
    std::shared_ptr<GraphBuffer> input_image = node->children_[node->arg_order_[0]]->output_;
    std::shared_ptr<GraphBuffer> output_image = node->output_;
    std::shared_ptr<GraphBuffer> kernel = node->children_[node->arg_order_[1]]->output_;

    int in = input_image->shape_.size();
    int on = output_image->shape_.size();
    int kn = kernel->shape_.size();

    int ix = input_image->shape_[in - 2];
    int iy = input_image->shape_[in - 3];
    int input_channels = input_image->shape_[in - 1];

    int ox = output_image->shape_[on - 2];
    int oy = output_image->shape_[on - 3];

    int kx = kernel->shape_[kn - 2];
    int ky = kernel->shape_[kn - 3];
    int kernel_channels = kernel->shape_[kn - 1];

    // kernel offsets
    int dk_x = std::floor(kx / 2.);
    int dk_y = std::floor(ky / 2.);

    int batches = 1;
    for (int i = 0; i < in - 3; i++) {
        batches *= input_image->shape_[i];
    }

    int input_image_size = 1;
    for (int i = in - 3; i < in; i++) {
        input_image_size *= input_image->shape_[i];
    }

    int output_image_size = 1;
    for (int i = on - 3; i < on; i++) {
        output_image_size *= output_image->shape_[i];
    }

    // 7 for loops lol
    for (int batch = 0; batch < batches; batch++) {
        // iterate over pixels in the output image
        for (int y = 0; y < oy; y++) {
            for (int x = 0; x < ox; x++) {
                float output_value = 0;

                // mapping from output coordinates to input coordinates
                int input_x = x + dk_x;
                int input_y = y + dk_y;

                // iterating over the kernel
                for (int kc = 0; kc < kernel_channels; kc++) {
                    for (int i = -std::ceil(ky / 2.) + 1; i < std::ceil(ky / 2.); i++) {
                        for (int j = -std::ceil(kx / 2.) + 1; j < std::ceil(kx / 2.); j++) {
                            int kernel_x = dk_x - j;
                            int kernel_y = dk_y - i;

                            // iterating over each channel
                            for (int ic = 0; ic < input_channels; ic++) {
                                float input_value =
                                    input_image->getIndex<float>(batch * input_image_size +             // batch index
                                                                 (input_y + i) * ix * input_channels +  // y index
                                                                 (input_x + j) * input_channels + ic);  // x index

                                float kernel_value = kernel->getIndex<float>(kernel_y * kx * kernel_channels +
                                                                             kernel_x * kernel_channels + kc);

                                output_value += input_value * kernel_value;
                            }

                            // output will be the average of each input channel
                            output_value /= input_channels;

                            output_image->setIndex(batch * output_image_size +     // batch index
                                                       y * ox * kernel_channels +  // y index
                                                       x * kernel_channels +       // x index
                                                       kc,                         // channel index
                                                   (void*)(&output_value));
                        }
                    }
                }
            }
        }
    }
}

}  // namespace kernel
