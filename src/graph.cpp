#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

// change this lol
#include "buffer.cpp"
#include "ops.cpp"

class Node;
class Graph;

class Node {
   public:
    Node(int id) : id_(id) {}

    int getId() {
        return id_;
    }

    void setInput(std::string name, std::shared_ptr<Buffer> buffer) {
        inputs_[name] = buffer;
    }

    void setOutput(std::string name, std::shared_ptr<Buffer> buffer) {
        outputs_[name] = buffer;
    }

   private:
    int id_;

    std::map<std::string, std::shared_ptr<Buffer>> inputs_;
    std::map<std::string, std::shared_ptr<Buffer>> outputs_;

    friend class Graph;
};

class Graph {
   public:
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

    // TODO: this probably needs moved out of this class
    void newTensor(std::string name, DTYPE dtype, std::vector<int> shape) {
        size_t total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }

        std::shared_ptr<Buffer> buffer(new Buffer(total_size, dtype));
        std::shared_ptr<Node> node = newNode();

        node->setOutput(name, buffer);
    }

    void createOperation(std::string op_name, std::string variable_name, std::vector<std::string> args) {
        // TODO: requiring tensor arguments to be constant integers
        //       setting variables for dimension sizes?
        //       this definitely needs to be changed
        if (op_name == Operations::TENSOR) {
            std::vector<int> tensor_args;
            for (std::string& s : args) {
                tensor_args.push_back(std::stoi(s));
            }

            newTensor(variable_name, DTYPE::float32, tensor_args);
        }
    }

   private:
    std::unordered_map<int, std::set<int>> edges_;          // id -> { neighbor_ids... }
    std::unordered_map<int, std::shared_ptr<Node>> nodes_;  // id -> Node*

    int node_index_ = 0;

    Graph() {}
};
