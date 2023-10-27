#include <algorithm>
#include <climits>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <utility>
#include <vector>

// change this lol
#include "buffer.cpp"
#include "ops.h"
#include "string_utils.cpp"

class Node;
class Graph;

template <typename T>
std::string vecToString(const std::vector<T>& vec) {
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i != vec.size() - 1) {
            oss << ", ";
        }
    }

    oss << "]";
    return oss.str();
}

class Node {
   public:
    Node(int id) : id_(id) {}

    int getId() { return id_; }

    std::string operation_type_;
    std::string name_;

    std::vector<std::string> arg_order_;
    std::map<std::string, std::shared_ptr<Node>> children_;

    std::map<std::string, std::shared_ptr<Buffer>> inputs_;
    std::shared_ptr<Buffer> output_;

    std::vector<int> shape_;

    void printOutput() {
        std::vector<int> indices(shape_.size(), 0);
        std::vector<int> end;
        for (int i : shape_) {
            end.push_back(i - 1);
        }

        while (indices != end) {
            int i = indices.size() - 1;
            for (int i = indices.size() - 1; i > 0 && indices[i] > end[i]; i--) {
                indices[i] = 0;
                indices[i - 1]++;

                std::cout << std::endl;
            }

            std::cout << output_->getIndex<float>(calculateIndex(indices)) << ", ";

            indices[indices.size() - 1]++;
        }

        std::cout << output_->getIndex<float>(calculateIndex(end)) << std::endl;
    }

   private:
    int id_;

    size_t calculateIndex(const std::vector<int>& indices) {
        if (indices.size() != shape_.size()) {
            std::cerr << "Node::calculateIndex error: indices.size() must be equal to shape_.size(). Got "
                      << indices.size() << " and " << shape_.size() << std::endl;
            exit(-1);
        }

        int shape_prod = 1;
        for (int i : shape_) {
            shape_prod *= i;
        }

        size_t final_index = 0;

        for (int i = 0; i < indices.size(); i++) {
            shape_prod /= shape_[i];
            final_index += indices[i] * shape_prod;
        }

        return final_index;
    }

    friend class Graph;
};

namespace allocation {

// TODO: shapes need figured out to a cleaner solution

// these functions allocate buffers for their given nodes
void allocateTensorNode(std::shared_ptr<Node> node) {
    size_t size = 1;
    for (const std::string& arg : node->arg_order_) {
        size *= std::stoi(arg);  // tensor() *should* have all numeric args if it's made it this far
    }

    node->output_ = std::shared_ptr<Buffer>(new Buffer(size, DTYPE::float32));
}

// NOTE: broadcasting is currently not supported
//       this means given inputs MUST be the same shape,
//       save for the last two dimensions
void allocateMatmulNode(std::shared_ptr<Node> node) {
    if (node->arg_order_.size() != 2) {
        std::cerr << "allocateMatmulNode error: matmul node has <> 2 args, how did this happen?" << std::endl;
        exit(-1);
    }

    size_t size = 1;
    std::vector<int> shape_a, shape_b;
    shape_a = node->children_[node->arg_order_[0]]->shape_;
    shape_b = node->children_[node->arg_order_[1]]->shape_;

    if (shape_a.size() != shape_b.size()) {
        std::cerr << "allocateMatmulNode error: argument shapes must be equal in size. Got " << vecToString(shape_a)
                  << " and " << vecToString(shape_b) << std::endl;
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

    size *= shape_a[n - 2] * shape_b[n - 1];

    node->output_ = std::shared_ptr<Buffer>(new Buffer(size, DTYPE::float32));
    node->shape_ = new_shape;
}

// all constants will be assumed to be 32-bit float values
void allocateConstantNode(std::shared_ptr<Node> node) {
    node->output_ = std::shared_ptr<Buffer>(new Buffer(1, DTYPE::float32));

    float value = std::stof(node->name_);
    node->output_->setIndex(0, (void*)(&value));
}

void allocateNode(std::shared_ptr<Node> node) {
    if (node->operation_type_ == Operations::TENSOR) {
        allocateTensorNode(node);
    } else if (node->operation_type_ == Operations::MATMUL) {
        allocateMatmulNode(node);
    } else if (node->operation_type_ == Operations::CONSTANT) {
        allocateConstantNode(node);
    }
}

}  // namespace allocation

std::string randomString(size_t length) {
    auto randchar = []() -> char {
        const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[rand() % max_index];
    };

    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
}

class Graph {
   public:
    Graph() {}

    std::shared_ptr<Node> newNode() {
        std::shared_ptr<Node> node(new Node(node_index_));
        node_index_++;

        nodes_[node->getId()] = node;

        return node;
    }

    // TODO: is there other processing we want to do here?
    void newEdge(int from, int to) {
        if (nodes_.find(from) == nodes_.end()) {
            std::cerr << "Graph::newEdge error: node id " << from << " not registered with graph" << std::endl;
            exit(-1);
        }

        if (nodes_.find(to) == nodes_.end()) {
            std::cerr << "Graph::newEdge error: node id " << to << " not registered with graph" << std::endl;
            exit(-1);
        }

        edges_[from].insert(to);
    }

    // variables must spawn from an operation
    // there is no declaration operation
    // e.g.
    //      let A = tensor(256, 512); // this is allowed
    //      let B;                    // this is not allowed
    //
    // NOTE: edges_ isn't being updated here
    //       is it needed as a field?
    std::string createVariable(const std::string& name, const std::string& operation_type,
                               const std::vector<std::string>& arguments) {
        std::shared_ptr<Node> new_node = newNode();

        new_node->name_ = name;
        new_node->operation_type_ = operation_type;
        new_node->arg_order_ = arguments;

        while (new_node->name_.size() == 0 || variable_map_.find(new_node->name_) != variable_map_.end()) {
            new_node->name_ = name + "_" + randomString(5);
        }

        variable_map_[new_node->name_] = new_node;

        for (const std::string& arg : arguments) {
            if (strings::isNumeric(arg)) {
                // create a Constant-type node
                // set as child
                int constant = stoi(arg);
                if (constant_map_.find(constant) == constant_map_.end()) {
                    createConstant(constant);
                }

                new_node->children_[arg] = constant_map_[constant];
                edges_[new_node->id_].insert(constant_map_[constant]->id_);

                // there's gotta be a better way to do this
                if (operation_type == Operations::TENSOR) {
                    new_node->shape_.push_back(constant);
                }
            } else if (variable_map_.find(alias_map_[arg]) != variable_map_.end()) {
                // set the pre-existing variable node as a child
                new_node->children_[arg] = variable_map_[alias_map_[arg]];
                edges_[new_node->id_].insert(variable_map_[alias_map_[arg]]->id_);
            } else {
                std::cerr << "Graph::createVariable error: argument is neither numeric constant nor existing variable"
                          << std::endl;

                exit(-1);
            }
        }

        alias_map_[name] = new_node->name_;

        std::cout << "<<== BEGINNING ALLOCATION OF NODE " << new_node->name_ << " WITH TYPE "
                  << new_node->operation_type_ << std::endl;
        allocation::allocateNode(new_node);
        std::cout << "ALLOCATED NODE " << new_node->name_ << " WITH SHAPE " << vecToString(new_node->shape_)
                  << std::endl;

        return new_node->name_;
    }

    void createConstant(int constant) {
        if (constant_map_.find(constant) != constant_map_.end()) {
            std::cerr << "Graph::createConstant error: Node for constant " << constant
                      << " already exists. Why is this being created again?" << std::endl;
        }

        std::shared_ptr<Node> constant_ptr = newNode();

        constant_ptr->name_ = std::to_string(constant);
        constant_ptr->operation_type_ = Operations::CONSTANT;

        constant_map_[constant] = constant_ptr;
    }

    bool isVariable(const std::string& name) { return variable_map_.find(alias_map_[name]) != variable_map_.end(); }

    void listNodes() {
        for (auto& p : variable_map_) {
            std::cout << p.first << " " << vecToString(p.second->shape_) << ": " << std::endl;
            for (const std::string& n : p.second->arg_order_) {
                std::cout << "  - " << n << std::endl;
            }

            std::cout << std::endl;
        }
    }

    void printNodeValues() {
        for (auto& p : variable_map_) {
            std::cout << p.first << " " << vecToString(p.second->shape_) << ": " << std::endl;
            p.second->printOutput();
            std::cout << std::endl;
        }
    }

    // bfs
    // assuming the graph is directed and acyclic
    void debugTraversal() {
        std::map<int, int> incoming_counts;
        for (auto& p : edges_) {
            for (int edge : p.second) {
                incoming_counts[edge]++;
            }
        }

        int root = 0;
        int min_edges = INT_MAX;
        for (auto& p : incoming_counts) {
            if (p.second < min_edges) {
                root = p.first;
                min_edges = p.second;
            }
        }

        std::queue<int> q;
        q.push(root);

        int level = 0;
        while (!q.empty()) {
            int n = q.size();

            for (int i = 0; i < n; i++) {
                std::shared_ptr<Node> current = nodes_[q.front()];
                q.pop();

                std::cout << "level, id, name, type: " << level << ", " << current->getId() << ", " << current->name_
                          << ", " << current->operation_type_ << std::endl;

                for (auto& p : current->children_) {
                    q.push(p.second->getId());
                }
            }
        }
    }

   private:
    std::map<int, std::set<int>> edges_;          // id -> { neighbor_ids... } outgoing edges
    std::map<int, std::shared_ptr<Node>> nodes_;  // id -> Node*

    std::map<int, std::shared_ptr<Node>> constant_map_;
    std::map<std::string, std::shared_ptr<Node>> variable_map_;

    // this is for when a variable is reassigned
    // e.g. let A = tensor(1, 2)
    //      A = tensor(3, 4)      // this will have a different alias in the graph
    std::map<std::string, std::string> alias_map_;

    int node_index_ = 0;
};
