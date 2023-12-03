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

// is this also where validation will be taking place?
namespace allocation {

// TODO: shapes need figured out to a cleaner solution
// TODO: input argument count validation

void _input_validator(size_t expected, size_t received, const std::string& operation) {
    if (expected != received) {
        std::cerr << strings::error("allocation::" + operation + " error: ") << strings::info(operation) << " expects "
                  << expected << " operands, received " << received << std::endl;
        exit(-1);
    }
}

// TODO: can this be reused between functions and network inputs?
void inputAllocate(std::shared_ptr<Node> node) {
    // is this an input to a function?
    // this doesn't actually allocate anything, but the connections are made between the relevant properties
    if (node->children_.size() > 0) {
        std::shared_ptr<Node> input_node = node->children_.begin()->second;

        node->shape_ = input_node->shape_;
        node->output_ = input_node->output_;
        node->gradient_ = input_node->gradient_;
    } else {
        // otherwise this needs allocated space
        size_t size = 1;
        for (int i : node->shape_) {
            size *= i;
        }

        node->output_ = std::shared_ptr<Buffer>(new Buffer(size, DTYPE::float32));
        // NOTE: leaving `gradient_` unallocated -- will this be an issue?
    }
}

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
    node->gradient_ = std::shared_ptr<Buffer>(new Buffer(size, DTYPE::float32));
    const float one = 1;
    for (size_t i = 0; i < node->gradient_->size(); i++) {
        node->gradient_->setIndex(i, (void*)(&one));
    }

    node->shape_ = shape;
}

// shape validators are called before this function happens
//
// this function is for allocating binary nodes that don't change shape as a result of their operation
// e.g. pemdas, pow, etc.
void _element_wise_allocate(std::shared_ptr<Node> node) {
    node->output_ =
        std::shared_ptr<Buffer>(new Buffer(node->children_[node->arg_order_[0]]->output_->size(), DTYPE::float32));
    node->gradient_ =
        std::shared_ptr<Buffer>(new Buffer(node->children_[node->arg_order_[0]]->output_->size(), DTYPE::float32));
    const float one = 1;
    for (size_t i = 0; i < node->gradient_->size(); i++) {
        node->gradient_->setIndex(i, (void*)(&one));
    }

    node->shape_ = node->children_[node->arg_order_[0]]->shape_;
}

void addAllocate(std::shared_ptr<Node> node) {
    _input_validator(2, node->arg_order_.size(), "add");
    _element_wise_allocate(node);
}

void subtractAllocate(std::shared_ptr<Node> node) {
    _input_validator(2, node->arg_order_.size(), "subtract");
    _element_wise_allocate(node);
}

void multiplyAllocate(std::shared_ptr<Node> node) {
    _input_validator(2, node->arg_order_.size(), "multiply");
    _element_wise_allocate(node);
}

void divideAllocate(std::shared_ptr<Node> node) {
    _input_validator(2, node->arg_order_.size(), "divide");
    _element_wise_allocate(node);
}

void sqrtAllocate(std::shared_ptr<Node> node) {
    _input_validator(1, node->arg_order_.size(), "sqrt");
    _element_wise_allocate(node);
}

void expAllocate(std::shared_ptr<Node> node) {
    _input_validator(1, node->arg_order_.size(), "exp");
    _element_wise_allocate(node);
}

void powAllocate(std::shared_ptr<Node> node) {
    _input_validator(2, node->arg_order_.size(), "pow");
    _element_wise_allocate(node);
}

// NOTE: broadcasting is currently not supported
//       this means given inputs MUST be the same shape,
//       save for the last two dimensions
void matmulAllocate(std::shared_ptr<Node> node) {
    if (node->arg_order_.size() != 2) {
        std::cerr << strings::error("matmulAllocateError: ") << "matmul node has <> 2 args, how did this happen?"
                  << std::endl;
        exit(-1);
    }

    size_t size = 1;
    std::vector<int> shape_a, shape_b;

    shape_a = node->children_[node->arg_order_[0]]->shape_;
    shape_b = node->children_[node->arg_order_[1]]->shape_;

    if (shape_a.size() != shape_b.size()) {
        std::cerr << strings::error("allocation::matmulAllocateError: ")
                  << "argument shapes must be equal in size. Got " << strings::vecToString(shape_a) << " and "
                  << strings::vecToString(shape_b) << std::endl;
        exit(-1);
    }

    int n = shape_a.size();
    for (int i = 0; i < n - 2; i++) {
        if (shape_a[i] != shape_b[i]) {
            std::cerr << strings::error("allocation::matmulAllocateError: ")
                      << "shapes must be equal until the final two dimensions [N - 2, N - 1]" << std::endl;
            exit(-1);
        }

        size *= shape_a[i];
    }

    if (shape_a.size() < 2 || shape_b.size() < 2) {
        node->printNode();

        std::cout << strings::error("INPUT NAMES: ") << strings::info(node->children_[node->arg_order_[0]]->name_)
                  << " and " << strings::info(node->children_[node->arg_order_[1]]->name_) << std::endl;

        node->children_[node->arg_order_[0]]->printNode();
        node->children_[node->arg_order_[1]]->printNode();

        std::cerr << strings::error("allocation::matmulAllocateError: ")
                  << "matmul shapes must be at least 2 dimensions in shape, got "
                  << strings::info(strings::vecToString(shape_a)) << " and "
                  << strings::info(strings::vecToString(shape_b)) << std::endl;
        exit(-1);
    }

    std::vector<int> new_shape = shape_a;

    new_shape[n - 2] = shape_a[n - 2];
    new_shape[n - 1] = shape_b[n - 1];

    if (shape_a[n - 1] != shape_b[n - 2]) {
        std::cerr << strings::error("allocation::matmulAllocate error: ")
                  << "incompatible shapes for matrix multiplication. Got "
                  << strings::info(strings::vecToString(shape_a)) << " and "
                  << strings::info(strings::vecToString(shape_b)) << std::endl;
        exit(-1);
    }

    size *= shape_a[n - 2] * shape_b[n - 1];

    node->output_ = std::shared_ptr<Buffer>(new Buffer(size, DTYPE::float32));
    node->gradient_ = std::shared_ptr<Buffer>(new Buffer(size, DTYPE::float32));
    const float one = 1;
    for (size_t i = 0; i < node->gradient_->size(); i++) {
        node->gradient_->setIndex(i, (void*)(&one));
    }

    node->shape_ = new_shape;
}

// all constants will be assumed to be 32-bit float values
// does gradient_ need allocated here?
// trivial memory usage either way
void constantAllocate(std::shared_ptr<Node> node) {
    node->output_ = std::shared_ptr<Buffer>(new Buffer(1, DTYPE::float32));
    node->gradient_ = std::shared_ptr<Buffer>(new Buffer(1, DTYPE::float32));

    float value = std::stof(node->name_);
    float zero = 0;

    node->output_->setIndex(0, (void*)(&value));
    node->gradient_->setIndex(0, (void*)(&zero));
}

void normalAllocate(std::shared_ptr<Node> node) {
    tensorAllocate(node);
    std::shared_ptr<Buffer> buf = node->output_;

    if (buf->dtype() != DTYPE::float32) {
        std::cerr << strings::error("allocation::allocateNormalNode error: ") << "buffer data type must be float32"
                  << std::endl;
        exit(-1);
    }

    // is a separate data buffer really necessary?
    std::vector<float> data(buf->size());
    generation::fillNormal(data);

    for (size_t i = 0; i < buf->size(); i++) {
        buf->setIndex(i, (void*)(&data[i]));
    }
}

void onesAllocate(std::shared_ptr<Node> node) {
    tensorAllocate(node);
    const float one = 1;
    for (size_t i = 0; i < node->output_->size(); i++) {
        node->output_->setIndex(i, (void*)(&one));
    }
}

void sigmoidAllocate(std::shared_ptr<Node> node) {
    _input_validator(1, node->arg_order_.size(), "sigmoid");
    _element_wise_allocate(node);
}

void reluAllocate(std::shared_ptr<Node> node) {
    _input_validator(1, node->arg_order_.size(), "relu");
    _element_wise_allocate(node);
}

void allocateNode(std::shared_ptr<Node> node) {
    const auto& allocationMap = OperationRegistry::GetAllocationMap();

    auto it = allocationMap.find(node->operation_type_);
    if (it != allocationMap.end()) {
        it->second(node);
    } else {
        std::cerr << strings::error("allocation::allocateNode error: ") << "unrecognized node operation type "
                  << strings::info("`" + node->operation_type_ + "`") << std::endl;
        exit(-1);
    }
}

}  // namespace allocation
