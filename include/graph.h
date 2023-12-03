#ifndef GRAPH
#define GRAPH

#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "buffer.h"

class Node;
class Graph;

// probably needs moved

// this class feels like it's getting too fat
class Node {
   public:
    Node(int id);

    Node(std::shared_ptr<Node> node);

    int getId();

    std::string operation_type_;
    std::string name_;

    std::vector<std::string> arg_order_;

    // inputs to the node
    std::map<std::string, std::shared_ptr<Node>> children_;

    // TODO: should i overload this class?
    // this is only used by input nodes
    // to figure out their shape during an allocation call
    std::shared_ptr<Node> input_mapping_;

    std::shared_ptr<Buffer> output_;
    std::shared_ptr<Buffer> gradient_;

    std::vector<int> shape_;

    // this is only used if operation_type_ == operations::function
    std::shared_ptr<Graph> graph_;

    // this is true if the node is an input to the `.nn` file; false otherwise
    bool external_input;

    void printOutput();

    void printGradient();

    void printNode();

   private:
    int id_;

    friend class Graph;
};

class Graph {
   public:
    Graph();

    Graph(std::shared_ptr<Graph> graph);

    std::string getUniqueNodeName(const std::string& name);

    std::shared_ptr<Node> newNode();

    // TODO: is there other processing we want to do here?
    void newEdge(int from, int to);

    // variables must spawn from an operation
    // there is no declaration operation
    // e.g.
    //      let A = tensor(256, 512); // this is allowed
    //      let B;                    // this is not allowed
    //
    // NOTE: edges_ isn't being updated here
    //       is it needed as a field?
    std::string createVariable(const std::string& name, const std::string& operation_type,
                               const std::vector<std::string>& arguments);

    std::string createFunctionVariable(const std::string& name, const std::vector<std::string>& arguments,
                                       const std::shared_ptr<Graph> graph);

    void createConstant(int constant);

    bool isVariable(const std::string& name);

    bool isNode(int id);

    std::shared_ptr<Node> getNode(int id);

    std::vector<std::shared_ptr<Node>> getInputs();

    void evaluate();

    void allocate();

    void print();

    // retrieve a map of the gradient of the head wrt every node in the graph
    std::unordered_map<std::string, std::shared_ptr<Node>> gradient();

    // topological sort for evaluate and allocate
    void topologicalSort(std::function<void(std::shared_ptr<Node>)> visit_function);

    // NOTE: this will probably have to change
    //       as it works only under the assumption
    //       that functions/graphs return only one output
    std::shared_ptr<Node> getHead();

    void listNodes();

    void printNodeValues();

    // this is the most reliable list of nodes in the graph
    std::map<int, std::shared_ptr<Node>> nodes_;  // id -> Node*

   private:
    // this isn't really used, do we need it?
    std::map<int, std::set<int>> edges_;  // id -> { neighbor_ids... } outgoing edges

    std::map<int, std::shared_ptr<Node>> constant_map_;
    std::map<std::string, std::shared_ptr<Node>> variable_map_;

    // container for used functions in the graph
    std::vector<std::shared_ptr<Graph>> subgraphs_;

    // keeping track of the input nodes without having to iterate the whole graph
    std::map<std::string, std::shared_ptr<Node>> inputs_;

    std::shared_ptr<Node> _create_variable(const std::string& name, const std::string& operation_type,
                                           const std::vector<std::string>& arguments);

    // this is for when a variable is reassigned
    // e.g. let A = tensor(1, 2)
    //      A = tensor(3, 4)      // this will have a different alias in the graph
    std::map<std::string, std::string> alias_map_;

    int node_index_ = 0;
};

#endif
