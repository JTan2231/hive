#include "graph.h"
#include "nn_parser.h"

using namespace std;

int main() {
    const string filepath = "./nn/function.nn";
    const string contents = nn_parser::readFile(filepath);

    nn_parser::NNParser parser(contents);
    std::shared_ptr<Graph> g = parser.parse(contents);
    g->print();
    g->allocate();
    // g->evaluate();
    // g->printNodeValues();
}
