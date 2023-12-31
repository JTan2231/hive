#include "graph.h"

#include <algorithm>
#include <climits>
#include <fstream>
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
#include "buffer_ops.h"
#include "grad.h"
#include "iterators.h"
#include "kernel.h"
#include "logging.h"
#include "ops.h"
#include "string_utils.h"

// TODO: there NEEDS to be some sort of differentiator between differentiable variables and constant numbers
//       this is probably a language problem. new keyword? const vs var?

Node::Node(int id) : id_(id), external_input_(false), trainable_(false) {
}

Node::Node(std::shared_ptr<Node> node)
    : id_(node->id_),
      operation_type_(node->operation_type_),
      name_(node->name_),
      arg_order_(node->arg_order_),
      shape_(node->shape_),
      trainable_(node->trainable_) {
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

// these two print functions can probably be abstracted

void Node::printOutput(std::ostream& stream) {
    if (shape_.size() < 2) {
        for (size_t i = 0; i < output_->size(); i++) {
            stream << output_->getIndex<float>(i) << ", ";
        }

        stream << std::endl;

        return;
    }

    std::vector<int> batch_shape;

    if (shape_.size() > 2) {
        for (int i = 0; i < shape_.size() - 2; i++) {
            batch_shape.push_back(shape_[i]);
        }
    } else {
        batch_shape = shape_;
    }

    iterators::IndexIterator it(batch_shape);

    int m = shape_[shape_.size() - 2];
    int n = shape_[shape_.size() - 1];

    stream << "    ";
    if (shape_.size() > 2) {
        while (!it.end()) {
            size_t index = it.getIndex();

            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    std::string output = std::to_string(output_->getIndex<float>(index + (r * n) + c));
                    stream << output << ", ";
                }
            }

            it.increment();
        }

        stream << std::endl;
    } else if (shape_.size() == 2) {
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                stream << output_->getIndex<float>(r * n + c) << ", ";
            }

            stream << std::endl;
        }
    }

    stream << std::endl;
}

void Node::printGradient(std::ostream& stream) {
    if (shape_.size() < 2) {
        for (size_t i = 0; i < gradient_->size(); i++) {
            stream << gradient_->getIndex<float>(i) << ", ";
        }

        stream << std::endl;

        return;
    }

    std::vector<int> batch_shape;

    if (shape_.size() > 2) {
        for (int i = 0; i < shape_.size() - 2; i++) {
            batch_shape.push_back(shape_[i]);
        }
    } else {
        batch_shape = shape_;
    }

    iterators::IndexIterator it(batch_shape);

    int m = shape_[shape_.size() - 2];
    int n = shape_[shape_.size() - 1];

    stream << "    ";
    if (shape_.size() > 2) {
        while (!it.end()) {
            size_t index = it.getIndex();

            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    std::string output = std::to_string(gradient_->getIndex<float>(index + (r * n) + c));
                    stream << output << ", ";
                }
            }

            it.increment();
        }

        stream << std::endl;
    } else if (shape_.size() == 2) {
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                stream << gradient_->getIndex<float>(r * n + c) << ", ";
            }

            stream << std::endl;
        }
    }

    stream << std::endl;
}

void Node::printNode() {
    std::cout << strings::debug("Node " + std::to_string(id_) + ":") << std::endl;
    std::cout << strings::debug("- Name: ") << strings::info(name_) << std::endl;
    std::cout << strings::debug("- Shape: ") << strings::info(strings::vecToString(shape_)) << std::endl;
    std::cout << strings::debug("- Trainable: ") << strings::info(trainable_ ? "true" : "false") << std::endl;
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

std::shared_ptr<Node> Graph::getNode(const std::string& name) {
    if (alias_map_.find(name) != alias_map_.end()) {
        if (variable_map_.find(alias_map_[name]) != variable_map_.end()) {
            return variable_map_[alias_map_[name]];
        } else {
            std::cerr << strings::error("Graph::getNode error: ") << "node " << strings::info(name) << " or "
                      << strings::info(alias_map_[name]) << "doesn't exist" << std::endl;
            exit(-1);
        }
    } else {
        std::cerr << strings::error("Graph::getNode error: ") << "node " << strings::info(name) << "doesn't exist"
                  << std::endl;
        exit(-1);
    }
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
                                              const std::vector<std::string>& arguments, bool trainable,
                                              bool is_const) {
    std::shared_ptr<Node> new_node = newNode();

    new_node->name_ = name;
    new_node->operation_type_ = operation_type;
    new_node->arg_order_ = arguments;
    new_node->trainable_ = trainable;
    new_node->const_ = is_const;

    while (new_node->name_.size() == 0 || variable_map_.find(new_node->name_) != variable_map_.end()) {
        new_node->name_ = name + "_" + strings::randomString(5);
    }

    variable_map_[new_node->name_] = new_node;

    // input nodes require a name (string) and a shape
    // formatted like:
    //   - let inp = input("my input", 256, 256, 3)
    // where:
    //   - name == "my input"
    //   - shape = (256, 256, 3)
    if (operation_type == operations::input) {
        if (arguments.size() < 2) {
            std::cerr << strings::error("Graph::_create_variable error: ") << "input " << strings::info(new_node->name_)
                      << " requires a shape and a name, e.g. "
                      << strings::debug("let x = input(\"my input name\", 1, 2, 4, 8) ") << "has shape "
                      << strings::info("(1, 2, 4, 8)") << std::endl;
            exit(-1);
        }

        std::string input_name = arguments[0];

        for (int i = 1; i < arguments.size(); i++) {
            if (!strings::isNumeric(arguments[i])) {
                std::cerr << strings::error("Graph::_create_variable error: ") << "input() shape values must be numeric"
                          << std::endl;
                exit(-1);
            }

            new_node->shape_.push_back(std::stoi(arguments[i]));
        }

        new_node->external_input_ = true;

        if (inputs_.find(input_name) != inputs_.end()) {
            std::cerr << strings::error("Graph::_create_variable error: ") << "input name " << strings::info(input_name)
                      << " already exists" << std::endl;
            exit(-1);
        }

        inputs_[input_name] = new_node;
    } else {
        for (int i = 0; i < arguments.size(); i++) {
            const std::string& arg = arguments[i];

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
                new_node->children_[alias_map_[arg]] = variable_map_[alias_map_[arg]];
                new_node->arg_order_[i] = alias_map_[arg];
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
                                  const std::vector<std::string>& arguments, bool trainable, bool is_const) {
    std::shared_ptr<Node> new_node = _create_variable(name, operation_type, arguments, trainable, is_const);

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

void Graph::setLossNode(const std::string& name) {
    loss_node_ = name;
}

float Graph::getLoss() {
    std::shared_ptr<Node> loss_node = getNode(loss_node_);

    float loss = 0;
    for (int i = 0; i < loss_node->output_->size(); i++) {
        loss += loss_node->output_->getIndex<float>(i);
    }

    return loss;
}

// created during function calls
// function graph is cloned then merged into the main graph
std::string Graph::createFunctionVariable(const std::string& name, const std::vector<std::string>& arguments,
                                          const std::shared_ptr<Graph> graph) {
    // i have no clue what's happening here
    std::unique_ptr<Graph> graph_copy(new Graph(graph));
    std::shared_ptr<Node> head = graph_copy->getHead();

    std::vector<std::shared_ptr<Node>> inputs(arguments.size());

    std::map<std::string, std::string> name_map;

    // register all subgraph nodes in the main graph
    for (auto& [id, node] : graph_copy->nodes_) {
        std::string old_name = node->name_;
        if (node != head) {
            node->name_ = getUniqueNodeName(node->name_);
        } else {
            node->name_ = name;
        }

        // node->id_ will always be < arguments.size() if the node is an input node
        if (node->operation_type_ == operations::input) {
            std::shared_ptr<Node> input_node = variable_map_[alias_map_[arguments[node->id_]]];
            inputs[node->id_] = node;
            inputs[node->id_]->children_[input_node->name_] = input_node;
        }

        node->id_ = node_index_;
        node_index_++;

        nodes_[node_index_] = node;
        variable_map_[node->name_] = node;
        alias_map_[node->name_] = node->name_;

        name_map[old_name] = node->name_;
    }

    for (auto& [id, node] : graph_copy->nodes_) {
        if (node->operation_type_ == operations::input) {
            continue;
        }

        std::map<std::string, std::shared_ptr<Node>> renamed_children;
        for (auto& [name, child] : node->children_) {
            if (name_map.find(name) == name_map.end() ||
                std::find(node->arg_order_.begin(), node->arg_order_.end(), name) == node->arg_order_.end()) {
                std::cerr << strings::error("Cannot find name ") << name << std::endl;
                exit(-1);
            }

            renamed_children[name_map[name]] = child;
            for (int i = 0; i < node->arg_order_.size(); i++) {
                if (node->arg_order_[i] == name) {
                    node->arg_order_[i] = name_map[name];
                }
            }
        }

        node->children_ = renamed_children;

        for (auto& arg : node->arg_order_) {
            if (!(node->children_.find(arg) != node->children_.end())) {
                std::cerr << strings::error("Graph::createFunctionVariable error: ") << "could not find mapping for "
                          << strings::info(arg) << std::endl;
                exit(-1);
            }
        }
    }

    if (arguments.size() != inputs.size()) {
        std::cerr << strings::error("Graph::createFunctionVariable error: ") << "expected "
                  << strings::info("arguments.size() == inputs.size()") << ", got "
                  << strings::info(std::to_string(arguments.size())) << " and "
                  << strings::info(std::to_string(inputs.size())) << std::endl;
        exit(-1);
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

void Graph::log(std::ofstream& log_file) {
    for (auto& [id, node] : nodes_) {
        log_file << node->name_ << " " << strings::vecToString(node->shape_) << std::endl;
        node->printOutput(log_file);
    }
}

void Graph::gradLog(std::ofstream& log_file) {
    for (auto& [id, node] : nodes_) {
        log_file << node->name_ << " " << strings::vecToString(node->shape_) << std::endl;
        node->printGradient(log_file);
    }
}

void Graph::evaluate() {
    if (inputs_.size() > 0) {
        std::cerr << strings::error("Graph::evaluate error: ") << "missing values for inputs" << std::endl;
        exit(-1);
    }

    topologicalSort(kernel::computeNode);
}

// TODO: this will need adjusted for batches
//       the input loading here ONLY accounts for 1-D values
//       no tensors or batched values yet
void Graph::evaluate(std::unordered_map<std::string, std::vector<float>> inputs) {
    for (auto& [name, value] : inputs) {
        if (inputs_.find(name) == inputs_.end()) {
            std::cerr << strings::error("Graph::evaluate error: ") << "input " << strings::info(name)
                      << " not found in Graph" << std::endl;
            exit(-1);
        }

        std::shared_ptr<Node> input_node = inputs_[name];
        for (int i = 0; i < value.size(); i++) {
            input_node->output_->setIndex(i, (void*)(&value[i]));
        }
    }

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

void Graph::serialize(const std::string& filepath) {
    std::ofstream file;
    file.open(filepath, std::ios::out | std::ios::app);

    file << "{";

    int variables = 0;
    for (auto [id, node] : nodes_) {
        variables += node->trainable_;
    }

    for (auto [id, node] : nodes_) {
        if (node->trainable_) {
            file << "\"" << node->name_ << "\": [";
            for (size_t i = 0; i < node->output_->size() - 1; i++) {
                file << node->output_->getIndex<float>(i) << ", ";
            }

            file << node->output_->getIndex<float>(node->output_->size() - 1) << "]" << (variables > 1 ? "," : "");
            variables--;
        }
    }

    file << "}";
    file.close();
}

// bfs to propagate the gradient calculation down from the head
void Graph::calculateGradient() {
    inverseTopologicalSort(gradient::propagateNode);
}

void Graph::inverseTopologicalSort(std::function<void(std::shared_ptr<Node>)> visit_function) {
    std::queue<std::shared_ptr<Node>> q;
    q.push(getNode(loss_node_));

    std::stringstream log_stream;

    std::shared_ptr<Node> current;
    while (!q.empty()) {
        current = q.front();
        q.pop();

        visit_function(current);

        if (current->trainable_) {
            log_stream << current->name_ << " " << strings::vecToString(current->shape_) << std::endl;
            current->printGradient(log_stream);
            INFO(log_stream.str());
            log_stream.flush();
        }

        for (auto& [name, child] : current->children_) {
            q.push(child);
        }
    }
}

std::unordered_map<std::string, std::shared_ptr<Node>> Graph::getGradient() {
    std::unordered_map<std::string, std::shared_ptr<Node>> gradient;
    for (auto& [id, node] : nodes_) {
        if (node->trainable_) {
            gradient[node->name_] = node;
        }
    }

    return gradient;
}

// naive SGD for now
// TODO: where's Adam?
void Graph::applyGradients(int batch_size, float learning_rate) {
    for (auto& [id, node] : nodes_) {
        if (node->trainable_) {
            buffer_ops::multiply(node->gradient_, learning_rate, node->gradient_);
            buffer_ops::divide(node->gradient_, batch_size, node->gradient_);
            buffer_ops::subtract(node->output_, node->gradient_, node->output_);
        }
    }
}

void Graph::reset() {
    for (auto& [id, node] : nodes_) {
        if (!node->trainable_ && node->operation_type_ != operations::constant && !node->const_) {
            buffer_ops::set(node->output_, 0.);
        }

        buffer_ops::set(node->gradient_, 1.);
    }
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
        p.second->printOutput(std::cout);
        std::cout << std::endl;
    }
}
