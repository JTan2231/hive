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
#include "ops.cpp"
#include "string_utils.cpp"

class Node;
class Graph;

class Node {
   public:
    Node(int id) : id_(id) {}

    int getId() { return id_; }

    std::string operation_type_;
    std::string name_;

    std::map<std::string, std::shared_ptr<Node>> children_;

    std::map<std::string, std::shared_ptr<Buffer>> inputs_;
    std::map<std::string, std::shared_ptr<Buffer>> outputs_;

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

    // variables must spawn from an operation
    // there is no declaration operation
    // e.g.
    //      let A = tensor(256, 512); // this is allowed
    //      let B;                    // this is not allowed
    std::string createVariable(const std::string& name, const std::string& operation_type,
                               const std::vector<std::string>& arguments) {
        std::shared_ptr<Node> new_node = newNode();

        new_node->name_ = name;
        new_node->operation_type_ = operation_type;

        variable_map_[name] = new_node;

        for (const std::string& arg : arguments) {
            if (strings::isNumeric(arg)) {
                // create a Constant-type node
                // set as child
                int constant = stoi(arg);
                if (constant_map_.find(constant) == constant_map_.end()) {
                    createConstant(constant);
                }

                new_node->children_[arg] = constant_map_[constant];
            } else if (variable_map_.find(arg) != variable_map_.end()) {
                // set the pre-existing variable node as a child
                new_node->children_[arg] = variable_map_[arg];
            } else {
                std::cerr << "Graph::createVariable error: argument is neither numeric constant nor existing variable"
                          << std::endl;
            }
        }

        return "";
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

    int node_index_ = 0;

    Graph() {}
};
