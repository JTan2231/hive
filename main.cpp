#include <fstream>
#include <memory>

#include "buffer_ops.h"
#include "data/csv.h"
#include "dtypes.h"
#include "generation_utils.h"
#include "graph.h"
#include "iterators.h"
#include "kernel.h"
#include "logging.h"
#include "metrics.h"
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
    g->serialize("./hive_weights.json");

    const string dataset_path = "/home/joey/Downloads/sine_data.csv";
    vector<string> inputs = {"t"};
    vector<string> outputs = {"sine_value"};

    CSVDataset dataset = CSVDataset(dataset_path, inputs, outputs);
    const int batch_size = 32;
    const float learning_rate = 0.01;

    auto loss_node = g->getNode("mse");

    auto pred_node = g->getNode("final_output");
    auto label_node = g->getNode("label");

    g->setLossNode("mse");

    ofstream log_file;
    log_file.open("./logs/train.log", std::ios::out | std::ios::app);

    ofstream grad_file;
    grad_file.open("./logs/grad.log", std::ios::out | std::ios::app);

    metrics::MeanAbsoluteError mae;
    metrics::Mean loss;

    int epochs = 0;
    for (int i = 0; i < 3200; i++) {
        auto data = dataset.sample(batch_size);

        g->evaluate(data);

        mae.update(pred_node->output_, label_node->output_);
        loss.update(loss_node->output_);

        g->calculateGradient();

        grad_file << "TRAIN STEP " << i + 1 << endl;
        g->gradLog(grad_file);
        grad_file << "----------------" << endl;

        g->applyGradients(batch_size, learning_rate);

        if (LOG) {
            // INFO("Step " + to_string(i + 1) + " loss: " + to_string(loss));
            g->log(log_file);
        }

        cout << strings::error("TRAINING STEP " + to_string(i + 1) + ", MAE: ")
             << strings::debug(to_string(mae.value())) << strings::error(", LOSS: ")
             << strings::debug(to_string(loss.value())) << endl;

        if ((i + 1) % 32 == 0) {
            cout << strings::debug("        EPOCH " + to_string((i + 1) / 32) + " LOSS, METRIC: ") << loss.value()
                 << ", " << mae.value() << endl;

            loss.reset();
            mae.reset();
        }

        g->reset();
    }

    cout << endl;
}

void iteratorTest() {
    iterators::BroadcastIterator bi({1, 32, 32}, {32, 32, 32});

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

void bufferOpsTest() {
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

void indexIteratorTest() {
    iterators::IndexIterator it({4, 32, 32, 3});

    while (!it.end()) {
        cout << strings::vecToString(it.getIndices()) << ", " << it.getIndex() << endl;
        it.increment();
    }
}

void reduceSumTest() {
    std::shared_ptr<Buffer> a(new Buffer({32, 32, 3}, DTYPE::float32));
    std::shared_ptr<Buffer> b(new Buffer({1, 32, 3}, DTYPE::float32));

    buffer_ops::set(a, 1);
    buffer_ops::reduceSum(a, b, {0});

    cout << "b: " << endl;
    for (int i = 0; i < 32 * 3; i++) {
        cout << b->getIndex<float>(i) << ", ";
    }

    cout << endl;
}

void basicBinaryOpGradientTest(const string& op_name) {
    const string filepath = "./nn/tests/" + op_name + ".nn";
    const string contents = nn_parser::readFile(filepath);

    nn_parser::NNParser parser(contents);
    std::shared_ptr<Graph> g = parser.parse(contents);
    g->allocate();

    g->evaluate();
    g->calculateGradient();

    auto weights = g->getNode("weights");

    cout << "weights " << endl;
    weights->printOutput(cout);
    weights->printGradient(cout);

    cout << "w " << endl;
    auto w = g->getNode("w");
    w->printOutput(cout);
}

void basicUnaryOpGradientTest(const string& op_name) {
    const string filepath = "./nn/tests/" + op_name + ".nn";
    const string contents = nn_parser::readFile(filepath);

    nn_parser::NNParser parser(contents);
    std::shared_ptr<Graph> g = parser.parse(contents);
    g->allocate();

    g->evaluate();
    g->calculateGradient();

    auto weights = g->getNode("weights");

    cout << "weights " << endl;
    weights->printOutput(cout);
    weights->printGradient(cout);

    cout << "output " << endl;
    auto w = g->getNode("output");
    w->printOutput(cout);
}

void denseLayerTest() {
    const string filepath = "./nn/tests/dense.nn";
    const string contents = nn_parser::readFile(filepath);

    nn_parser::NNParser parser(contents);
    std::shared_ptr<Graph> g = parser.parse(contents);
    g->allocate();

    for (int i = 0; i < 10; i++) {
        g->evaluate();
        g->calculateGradient();
        g->applyGradients(1, 0.01);
        g->reset();
    }

    auto weights = g->getNode("weights");

    cout << "weights " << endl;
    weights->printOutput(cout);
    weights->printGradient(cout);

    cout << "i " << endl;
    auto i = g->getNode("i");
    i->printOutput(cout);
}

int main() {
    basicTrainingLoopTest();
}
