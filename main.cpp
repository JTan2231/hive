#include "graph.h"
#include "nn_parser.h"

using namespace std;

int main() {
    const string filepath = "./nn/function.nn";
    const string contents = nn_parser::readFile(filepath);

    nn_parser::NNParser parser(contents);
    Graph g = parser.parse(contents);
    g.evaluate();
    g.printNodeValues();
}
