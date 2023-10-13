#include <iostream>
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
   public:
    Node(int id) : id_(id) {}

    int getId() {
        return id_;
    }

   private:
    int id_;

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

   private:
    std::unordered_map<int, std::set<int>> edges_;          // id -> { neighbor_ids... }
    std::unordered_map<int, std::shared_ptr<Node>> nodes_;  // id -> Node*

    int node_index_ = 0;

    Graph() {}
};
