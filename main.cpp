#include <memory>

#include "data/csv.h"
#include "graph.h"
#include "nn_parser.h"
#include "string_utils.h"

using namespace std;

void graphTest() {
    const string filepath = "./nn/mlp.nn";
    const string contents = nn_parser::readFile(filepath);

    nn_parser::NNParser parser(contents);
    std::shared_ptr<Graph> g = parser.parse(contents);
    g->print();
}

void csvTest() {
    const string filepath = "./data/sine.csv";
    vector<string> inputs = {"t"};
    vector<string> outputs = {"sine_value"};

    CSVDataset dataset = CSVDataset(filepath, inputs, outputs);
    for (int i = 0; i < 10; i++) {
        auto sample = dataset.sample();
        cout << strings::error("TODO") << endl;
    }
}

void evaluateTest() {
    const string model_path = "./nn/mlp.nn";
    const string contents = nn_parser::readFile(model_path);

    nn_parser::NNParser parser(contents);
    std::shared_ptr<Graph> g = parser.parse(contents);
    g->allocate();

    const string dataset_path = "./data/sine.csv";
    vector<string> inputs = {"t"};
    vector<string> outputs = {"sine_value"};

    CSVDataset dataset = CSVDataset(dataset_path, inputs, outputs);

    g->evaluate(dataset.sample());
    g->printNodeValues();
}

int main() {
    evaluateTest();
}
