#include "allocation.h"

#include <memory>
#include <vector>

#include "buffer.h"
#include "generation_utils.h"
#include "graph.h"
#include "ops.h"
#include "string_utils.h"

// TODO: where should we put this?
template <typename T>
void printVec(const std::string& name, std::vector<T> v) {
    if (name.size() > 0) {
        std::cout << name << ": " << std::endl;
    }

    for (T& i : v) {
        std::cout << i << std::endl;
    }
}

namespace allocation {

// TODO: shapes need figured out to a cleaner solution

// these functions allocate buffers for their given nodes
void tensorAllocate(std::shared_ptr<Node> node) {
    size_t size = 1;
    std::vector<int> shape;
    for (const std::string& arg : node->arg_order_) {
        int dim = std::stoi(arg);  // tensor() *should* have all numeric args if it's made it this far

        size *= dim;
        shape.push_back(dim);
    }

    node->output_ = std::shared_ptr<Buffer>(new Buffer(size, DTYPE::float32));
    node->shape_ = shape;
}

// NOTE: broadcasting is currently not supported
//       this means given inputs MUST be the same shape,
//       save for the last two dimensions
void matmulAllocate(std::shared_ptr<Node> node) {
    if (node->arg_order_.size() != 2) {
        std::cerr << "allocateMatmulNode error: matmul node has <> 2 args, how did this happen?" << std::endl;
        exit(-1);
    }

    size_t size = 1;
    std::vector<int> shape_a, shape_b;

    shape_a = node->children_[node->arg_order_[0]]->shape_;
    shape_b = node->children_[node->arg_order_[1]]->shape_;

    if (shape_a.size() != shape_b.size()) {
        std::cerr << "allocateMatmulNode error: argument shapes must be equal in size. Got "
                  << strings::vecToString(shape_a) << " and " << strings::vecToString(shape_b) << std::endl;
        exit(-1);
    }

    int n = shape_a.size();
    for (int i = 0; i < n - 2; i++) {
        if (shape_a[i] != shape_b[i]) {
            std::cerr << "allocateMatmulNode error: shapes must be equal until the final two dimensions [N - 2, N - 1]"
                      << std::endl;
            exit(-1);
        }

        size *= shape_a[i];
    }

    std::vector<int> new_shape = shape_a;

    new_shape[n - 2] = shape_a[n - 2];
    new_shape[n - 1] = shape_b[n - 1];

    if (shape_a[n - 1] != shape_b[n - 2]) {
        std::cerr << "allocatedMatmulNode error: incompatible shapes for matrix multiplication. Got "
                  << strings::vecToString(shape_a) << " and " << strings::vecToString(shape_b) << std::endl;
        exit(-1);
    }

    size *= shape_a[n - 2] * shape_b[n - 1];

    node->output_ = std::shared_ptr<Buffer>(new Buffer(size, DTYPE::float32));
    node->shape_ = new_shape;
}

// all constants will be assumed to be 32-bit float values
void constantAllocate(std::shared_ptr<Node> node) {
    node->output_ = std::shared_ptr<Buffer>(new Buffer(1, DTYPE::float32));

    float value = std::stof(node->name_);
    node->output_->setIndex(0, (void*)(&value));
}

void normalAllocate(std::shared_ptr<Node> node) {
    tensorAllocate(node);
    std::shared_ptr<Buffer> buf = node->output_;

    if (buf->dtype() != DTYPE::float32) {
        std::cerr << "allocation::allocateNormalNode error: buffer data type must be float32" << std::endl;
        exit(-1);
    }

    std::vector<float> data(buf->size());
    generation::fillNormal(data);

    for (size_t i = 0; i < buf->size(); i++) {
        buf->setIndex(i, (void*)(&data[i]));
    }
}

void sigmoidAllocate(std::shared_ptr<Node> node) {
    size_t size = 1;
    std::vector<int> shape = node->children_[node->arg_order_[0]]->shape_;
    for (int i : shape) {
        size *= i;
    }

    node->output_ = std::shared_ptr<Buffer>(new Buffer(size, DTYPE::float32));
    node->shape_ = shape;
}

void reluAllocate(std::shared_ptr<Node> node) {
    sigmoidAllocate(node);
}

void allocateNode(std::shared_ptr<Node> node) {
    const auto& allocationMap = OperationRegistry::GetAllocationMap();

    auto it = allocationMap.find(node->operation_type_);
    if (it != allocationMap.end()) {
        it->second(node);
    } else {
        std::cerr << "allocation::allocateNode error: unrecognized node operation type " << node->operation_type_
                  << std::endl;
        exit(-1);
    }
}

}  // namespace allocation
