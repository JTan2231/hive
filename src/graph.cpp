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
#include "kernel.h"
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

        std::cout << output_->getIndex<float>(calculateIndex(indices, shape_)) << ", ";

        indices[indices.size() - 1]++;
    }

    std::cout << output_->getIndex<float>(calculateIndex(end, shape_)) << std::endl;
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
        // TODO: will constant args always mean it's a tensor op?
        if (strings::isNumeric(arg)) {
            // create a Constant-type node
            // set as child
            int constant = stoi(arg);
            if (constant_map_.find(constant) == constant_map_.end()) {
                createConstant(constant);
            }

            new_node->children_[arg] = constant_map_[constant];
            edges_[new_node->id_].insert(constant_map_[constant]->id_);

            new_node->shape_.push_back(constant);
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
    constant_ptr->operation_type_ = operations::constant;

    constant_map_[constant] = constant_ptr;
}

bool Graph::isVariable(const std::string& name) {
    return variable_map_.find(alias_map_[name]) != variable_map_.end();
}

// topological sort
// the number of edges a node has is determined by how many of its children's subgraphs are fully evaluated
// e.g. if a connected node has a fully calculated value, its edge is not taken into consideration
//      for topological sort
void Graph::evaluate() {
    std::map<std::string, int> degrees;
    std::queue<std::shared_ptr<Node>> q;

    std::map<std::string, std::set<std::string>> dependency_map;

    for (const auto& p : nodes_) {
        degrees[p.second->name_] = p.second->children_.size();
        if (p.second->children_.empty()) {
            q.push(p.second);
        }

        // have to invert the graph
        // for a bottom-up evaluation
        for (auto& cp : p.second->children_) {
            dependency_map[cp.first].insert(p.second->name_);
        }
    }

    std::shared_ptr<Node> current;
    while (!q.empty()) {
        current = q.front();
        q.pop();

        kernel::computeNode(current);

        for (const auto& dependent : dependency_map[current->name_]) {
            degrees[dependent]--;

            if (degrees[dependent] == 0) {
                // constants have no incoming edges
                // so this will *always* be a variable
                q.push(variable_map_[dependent]);
            }
        }
    }
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
