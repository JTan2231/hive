#include "graph.h"

#include <algorithm>
#include <climits>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "allocation.h"
#include "buffer.h"
#include "ops.h"
#include "string_utils.h"

Node::Node(int id) : id_(id) {}

int Node::getId() {
    return id_;
}

void Node::printOutput() {
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

size_t Node::calculateIndex(const std::vector<int>& indices) {
    if (indices.size() != shape_.size()) {
        std::cerr << "Node::calculateIndex error: indices.size() must be equal to shape_.size(). Got " << indices.size()
                  << " and " << shape_.size() << std::endl;
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

Graph::Graph() {}

std::shared_ptr<Node> Graph::newNode() {
    std::shared_ptr<Node> node(new Node(node_index_));
    node_index_++;

    nodes_[node->getId()] = node;

    return node;
}

// TODO: is there other processing we want to do here?
void Graph::newEdge(int from, int to) {
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
std::string Graph::createVariable(const std::string& name, const std::string& operation_type,
                                  const std::vector<std::string>& arguments) {
    std::shared_ptr<Node> new_node = newNode();

    new_node->name_ = name;
    new_node->operation_type_ = operation_type;
    new_node->arg_order_ = arguments;

    while (new_node->name_.size() == 0 || variable_map_.find(new_node->name_) != variable_map_.end()) {
        new_node->name_ = name + "_" + strings::randomString(5);
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

    allocation::allocateNode(new_node);

    return new_node->name_;
}

void Graph::createConstant(int constant) {
    if (constant_map_.find(constant) != constant_map_.end()) {
        std::cerr << "Graph::createConstant error: Node for constant " << constant
                  << " already exists. Why is this being created again?" << std::endl;
    }

    std::shared_ptr<Node> constant_ptr = newNode();

    constant_ptr->name_ = std::to_string(constant);
    constant_ptr->operation_type_ = Operations::CONSTANT;

    constant_map_[constant] = constant_ptr;
}

bool Graph::isVariable(const std::string& name) {
    return variable_map_.find(alias_map_[name]) != variable_map_.end();
}

void Graph::listNodes() {
    for (auto& p : variable_map_) {
        std::cout << p.first << " " << strings::vecToString(p.second->shape_) << ": " << std::endl;
        for (const std::string& n : p.second->arg_order_) {
            std::cout << "  - " << n << std::endl;
        }

        std::cout << std::endl;
    }
}

void Graph::printNodeValues() {
    for (auto& p : variable_map_) {
        std::cout << p.first << " " << strings::vecToString(p.second->shape_) << ": " << std::endl;
        p.second->printOutput();
        std::cout << std::endl;
    }
}

// bfs
// assuming the graph is directed and acyclic
void Graph::debugTraversal() {
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
