#include <fstream>
#include <memory>

#include "buffer_ops.h"
#include "data/csv.h"
#include "dtypes.h"
#include "generation_utils.h"
#include "graph.h"
#include "kernel.h"
#include "logging.h"
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
    const bool LOG = true;

    const string model_path = "./nn/mlp.nn";
    const string contents = nn_parser::readFile(model_path);

    nn_parser::NNParser parser(contents);
    std::shared_ptr<Graph> g = parser.parse(contents);
    g->allocate();

    const string dataset_path = "/home/joey/Downloads/scaled_synthetic_regression_data.csv";
    vector<string> inputs = {"Feature"};
    vector<string> outputs = {"Target"};

    CSVDataset dataset = CSVDataset(dataset_path, inputs, outputs);
    const int batch_size = 32;
    const float learning_rate = 0.001;

    auto head = g->getNode("mse");
    auto output = g->getNode("output");

    ofstream log_file;
    log_file.open("./logs/train.log", std::ios::out | std::ios::app);

    ofstream grad_file;
    grad_file.open("./logs/grad.log", std::ios::out | std::ios::app);

    float loss = 0;
    for (int i = 0; i < 1000; i++) {
        g->evaluate(dataset.sample(batch_size));
        g->calculateGradient();

        /*auto grads = g->getGradient();
        for (auto [name, node] : grads) {
            std::cout << name << std::endl;
            node->printGradient(std::cout);
        }*/

        g->applyGradients(batch_size, learning_rate);

        if (LOG) {
            INFO("Step " + to_string(i + 1) + " loss: " + to_string(loss));
            g->log(log_file);
        }

        float avg = 0;
        for (int i = 0; i < output->output_->size(); i++) {
            avg += output->output_->getIndex<float>(i);
        }

        avg /= output->output_->size();

        cout << strings::error("TRAINING STEP " + to_string(i + 1) + ", LOSS: ") << strings::debug(to_string(loss))
             << strings::error(", OUTPUT: ") << strings::debug(to_string(avg)) << endl;

        if (isnan(output->output_->getIndex<float>(0)) || isinf(output->output_->getIndex<float>(0))) {
            cout << strings::error("ERROR: NAN PREDICTION VALUE") << endl;
            exit(-1);
        }

        if (isnan(head->output_->getIndex<float>(0)) || isinf(head->output_->getIndex<float>(0))) {
            cout << strings::error("ERROR: NAN LOSS VALUE") << endl;
            exit(-1);
        } else {
            loss = 0;
            for (int i = 0; i < head->output_->size(); i++) {
                loss += head->output_->getIndex<float>(i);
            }

            loss /= head->output_->size();
        }
    }

    cout << endl;
}

void iteratorTest() {
    kernel::BroadcastIterator bi({1, 32, 32}, {32, 32, 32});

    cout << "initialized with " << endl;
    bi.print();
    cout << endl;

    while (bi.increment()) {
        bi.print();
        auto [lesser, greater] = bi.getIndices();
        cout << "- " << to_string(greater) << endl;
        cout << "- " << to_string(lesser) << endl;
        cout << endl;
    }
}

void betterIteratorTest() {
    vector<int> shape_a = {4, 5, 5};
    vector<int> shape_b = {1, 1, 1};

    std::shared_ptr<Buffer> A(new Buffer(shape_a, DTYPE::float32));
    std::shared_ptr<Buffer> B(new Buffer(shape_b, DTYPE::float32));
    std::shared_ptr<Buffer> C(new Buffer(shape_a, DTYPE::float32));

    generation::fillNormal(A);
    generation::fillNormal(B);

    A->print();
    B->print();

    buffer_ops::add(A, B, C);

    cout << strings::debug("C") << endl;
    for (int b = 0; b < 4; b++) {
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 5; c++) {
                cout << C->getIndex<float>(b * 25 + r * 5 + c) << ", ";
            }

            cout << endl;
        }

        cout << endl;
    }
}

int main() {
    basicTrainingLoopTest();
}
