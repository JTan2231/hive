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
        auto sample = dataset.sample(1);
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

    g->evaluate(dataset.sample(1));
    g->calculateGradient();
}

void basicTrainingLoopTest() {
    const string model_path = "./nn/mlp.nn";
    const string contents = nn_parser::readFile(model_path);

    nn_parser::NNParser parser(contents);
    std::shared_ptr<Graph> g = parser.parse(contents);
    g->allocate();

    const string dataset_path = "./data/sine.csv";
    vector<string> inputs = {"t"};
    vector<string> outputs = {"sine_value"};

    CSVDataset dataset = CSVDataset(dataset_path, inputs, outputs);
    const int batch_size = 32;

    auto head = g->getHead();

    float total_loss = 0;

    for (int i = 0; i < 1000; i++) {
        cout << "\r" << strings::error("TRAINING STEP " + to_string(i + 1) + ", LOSS: ")
             << strings::debug(to_string(total_loss / (i + 1))) << '\r';

        if (isnan(head->output_->getIndex<float>(0))) {
            cout << strings::error("ERROR: NAN LOSS VALUE") << endl;
            exit(-1);
        } else {
            float loss = 0;
            for (int i = 0; i < batch_size; i++) {
                total_loss += head->output_->getIndex<float>(i);
            }
        }

        g->evaluate(dataset.sample(batch_size));
        g->calculateGradient();
        g->applyGradients();
    }

    cout << endl;
}

int main() {
    basicTrainingLoopTest();
}
