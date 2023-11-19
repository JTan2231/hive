#include <memory>

#include "graph.h"
#include "nn_parser.h"
#include "string_utils.h"

using namespace std;

int main() {
    const string filepath = "./nn/grad.nn";
    const string contents = nn_parser::readFile(filepath);

    nn_parser::NNParser parser(contents);
    std::shared_ptr<Graph> g = parser.parse(contents);
    // g->print();
    g->allocate();
    g->evaluate();
    g->printNodeValues();

    std::unordered_map<std::string, std::shared_ptr<Node>> gradients = g->gradient();
    for (auto& [name, node] : gradients) {
        std::cout << strings::info("grad " + name + ":") << std::endl;
        node->printGradient();
        std::cout << std::endl;
    }
}
