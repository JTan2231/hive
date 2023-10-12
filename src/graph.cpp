#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

// change this lol
#include "buffer.cpp"
#include "ops/op.cpp"

class Node;
class Graph;

class Node {
    int id_;
    Operation* op_;

    Operation* getOp() { return op_; }

    void setId(int id) { id_ = id; }

    friend class Graph;
};

class Graph {
    std::unordered_map<int, std::set<int>> edges_;
    std::unordered_map<int, std::unique_ptr<Node>> nodes_;

    int node_index_ = 0;

    Graph() {}
};
