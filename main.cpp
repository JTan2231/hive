#include "nn_parser.h"

using namespace std;

int main() {
    const string filepath = "./nn/llm.nn";
    const string contents = nn_parser::readFile(filepath);

    nn_parser::NNParser parser(contents);
    parser.parse(contents);
}
