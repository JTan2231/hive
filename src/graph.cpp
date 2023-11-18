#include "graph.h"

#include <algorithm>
#include <climits>
#include <functional>
#include <iomanip>
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

// TODO: there NEEDS to be some sort of differentiator between differentiable variables and constant numbers
//       this is probably a language problem. new keyword? const vs var?

Node::Node(int id) : id_(id) {
}

Node::Node(std::shared_ptr<Node> node)
    : id_(node->id_),
      operation_type_(node->operation_type_),
      name_(node->name_),
      arg_order_(node->arg_order_),
      shape_(node->shape_) {
}

int Node::getId() {
    return id_;
}

std::string _node_format(float x) {
    constexpr int precision = 6;
    constexpr int width = precision + 4;  // decimal point + integer part (assuming 2 digits) + sign

    std::stringstream stream;
    stream << std::fixed << std::setw(width) << std::setprecision(precision) << x;

    return stream.str();
}

void Node::printOutput() {
    std::vector<int> indices(shape_.size(), 0);
    std::vector<int> end;
    for (int i : shape_) {
        end.push_back(i - 1);
    }

    std::cout << "    ";
    while (indices != end) {
        int i = indices.size() - 1;
        for (int i = indices.size() - 1; i > 0 && indices[i] > end[i]; i--) {
            indices[i] = 0;
            indices[i - 1]++;

            std::cout << std::endl << "    ";
        }

        std::string output = _node_format(output_->getIndex<float>(calculateIndex(indices, shape_)));

        std::cout << strings::debug(output + ", ");

        indices[indices.size() - 1]++;
    }

    std::cout << strings::debug(_node_format(output_->getIndex<float>(calculateIndex(end, shape_)))) << std::endl;
}

void Node::printNode() {
    std::cout << strings::debug("Node " + std::to_string(id_) + ":") << std::endl;
    std::cout << strings::debug("- Name: ") << strings::info(name_) << std::endl;
    std::cout << strings::debug("- Shape: ") << strings::info(strings::vecToString(shape_)) << std::endl;
    std::cout << strings::debug("- Children: ") << std::endl;
    for (auto& [name, child] : children_) {
        std::cout << strings::debug("  - ") << strings::info(name) << std::endl;
    }
}

Graph::Graph() {
}

Graph::Graph(std::shared_ptr<Graph> graph) {
    for (auto& [id, node] : graph->nodes_) {
        nodes_[id] = std::shared_ptr<Node>(new Node(node));
    }

    // copy children_ pointers
    for (auto& [id, node] : graph->nodes_) {
        std::shared_ptr<Node> copy_node = nodes_[id];
        for (auto& [name, child] : node->children_) {
            copy_node->children_[name] = nodes_[child->getId()];
        }
    }

    // copying over variable_map_
    for (auto& [name, node] : graph->variable_map_) {
        variable_map_[name] = nodes_[node->getId()];
    }

    alias_map_ = graph->alias_map_;
}

std::string Graph::getUniqueNodeName(const std::string& name) {
    std::string unique_name = name;
    while (variable_map_.find(unique_name) != variable_map_.end()) {
        unique_name = name + "_" + strings::randomString(5);
    }

    return unique_name;
}

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

std::shared_ptr<Node> Graph::_create_variable(const std::string& name, const std::string& operation_type,
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
        // TODO: this alias map nonsense is a headache
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
        } else if (variable_map_.find(arg) != variable_map_.end()) {
            // set the pre-existing variable node as a child
            new_node->children_[arg] = variable_map_[arg];
            edges_[new_node->id_].insert(variable_map_[arg]->id_);
        } else {
            std::cerr << strings::error("Graph::createVariable error: ")
                      << "argument " + strings::info("`" + arg + "`") +
                             " is neither numeric constant nor existing variable"
                      << std::endl;

            exit(-1);
        }
    }

    alias_map_[name] = new_node->name_;

    return new_node;
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
    std::shared_ptr<Node> new_node = _create_variable(name, operation_type, arguments);

    return new_node->name_;
}

bool Graph::isNode(int id) {
    return nodes_.find(id) != nodes_.end();
}

std::shared_ptr<Node> Graph::getNode(int id) {
    if (!isNode(id)) {
        std::cerr << strings::error("Graoh::getNode error: ") << "node with id " << id << " does not exist"
                  << std::endl;

        exit(-1);
    }

    return nodes_.find(id)->second;
}

std::vector<std::shared_ptr<Node>> Graph::getInputs() {
    std::vector<std::shared_ptr<Node>> inputs;
    for (auto& [id, node] : nodes_) {
        if (node->operation_type_ == operations::input) {
            inputs.push_back(node);
        }
    }

    return inputs;
}

// this is really gross
// one single node in the original graph that points to a separate graph
// which contains the function
//
// can this separate graph just be merged into the original graph?
std::string Graph::createFunctionVariable(const std::string& name, const std::vector<std::string>& arguments,
                                          const std::shared_ptr<Graph> graph) {
    // i have no clue what's happening here
    std::unique_ptr<Graph> graph_copy(new Graph(graph));
    std::shared_ptr<Node> head = graph_copy->getHead();

    // register all subgraph nodes in the main graph
    for (auto& [id, node] : graph_copy->nodes_) {
        if (node != head) {
            node->name_ = getUniqueNodeName(node->name_);
        } else {
            std::cout << name << " vs " << getUniqueNodeName(name) << std::endl;
            node->name_ = name;
        }

        nodes_[id] = node;
        variable_map_[node->name_] = node;
        alias_map_[node->name_] = node->name_;
    }

    std::vector<std::shared_ptr<Node>> inputs;
    int _id = 0;

    // inputs will ALWAYS (I think?) be the first nodes registered on the graph
    // this grabs the first N nodes in the graph_copy with the above assumption that they're inputs
    while (graph_copy->isNode(_id) && graph_copy->getNode(_id)->operation_type_ == operations::input) {
        inputs.push_back(graph_copy->getNode(_id));
        _id++;
    }

    if (arguments.size() != inputs.size()) {
        std::cerr << strings::error("Graph::createFunctionVariable error: ") << "expected "
                  << strings::info("arguments.size() == inputs.size()") << ", got "
                  << strings::info(std::to_string(arguments.size())) << " and "
                  << strings::info(std::to_string(inputs.size())) << std::endl;
        exit(-1);
    }

    // map each input node -> related argument node
    for (int i = 0; i < inputs.size(); i++) {
        std::shared_ptr<Node> input_node = variable_map_[alias_map_[arguments[i]]];
        inputs[i]->children_[input_node->name_] = input_node;
    }

    return head->name_;
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

void Graph::evaluate() {
    topologicalSort(kernel::computeNode);
}

void Graph::allocate() {
    topologicalSort(allocation::allocateNode);
}

void _print_node(std::shared_ptr<Node> node) {
    node->printNode();
}

void Graph::print() {
    topologicalSort(_print_node);
}

// topological sort
// the number of edges a node has is determined by how many of its children's subgraphs are fully evaluated
// e.g. if a connected node has a fully calculated value, its edge is not taken into consideration
//      for topological sort
void Graph::topologicalSort(std::function<void(std::shared_ptr<Node>)> visit_function) {
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

        std::cout << strings::error("VISITING NODE TYPE: " + current->operation_type_) << std::endl;

        visit_function(current);

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

// essentially a rpeeat of the above
std::shared_ptr<Node> Graph::getHead() {
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

        for (const auto& dependent : dependency_map[current->name_]) {
            degrees[dependent]--;

            if (degrees[dependent] == 0) {
                // constants have no incoming edges
                // so this will *always* be a variable
                q.push(variable_map_[dependent]);
            }
        }
    }

    return current;
}

void Graph::listNodes() {
    for (auto& p : variable_map_) {
        std::cout << strings::info(p.first + " " + strings::vecToString(p.second->shape_)) << ": " << std::endl;
        std::cout << "  - " << strings::debug("operation type: ") << p.second->operation_type_ << std::endl;

        std::cout << "  - " << strings::debug("inputs: ") << std::endl;
        for (const std::string& n : p.second->arg_order_) {
            std::cout << "    - " << strings::info(n) << std::endl;
        }

        std::cout << std::endl;
    }
}

void Graph::printNodeValues() {
    for (auto& p : variable_map_) {
        std::cout << strings::info(p.first + " " + strings::vecToString(p.second->shape_) + ": ") << std::endl;
        p.second->printOutput();
        std::cout << std::endl;
    }
}
