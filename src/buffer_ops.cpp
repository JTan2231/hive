#include "buffer_ops.h"

#include <memory>

#include "broadcasting.h"
#include "buffer.h"
#include "kernel.h"
#include "string_utils.h"

// weird mix of the kernel element-wise functions? this needs better organized
namespace buffer_ops {

void _assert_equal_sizes(const std::string& op, std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b,
                         std::shared_ptr<Buffer> c) {
    if (!kernel::broadcastable(a, b) && (a->size() != b->size() || a->size() != c->size())) {
        std::cerr << strings::error("buffer_ops::" + op + " error: ") << "input and output sizes must be equal, got "
                  << strings::info(std::to_string(a->size())) << ", " << strings::info(std::to_string(b->size()))
                  << ", and " << strings::info(std::to_string(c->size())) << std::endl;
        exit(-1);
    }
}

void _assert_equal_sizes(const std::string& op, std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b) {
    if (a->size() != b->size()) {
        std::cerr << strings::error("buffer_ops::" + op + " error: ") << "input and output sizes must be equal, got "
                  << strings::info(std::to_string(a->size())) << " and " << strings::info(std::to_string(b->size()))
                  << std::endl;
        exit(-1);
    }
}

void _assert_equal_dtypes(const std::string& op, std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b,
                          std::shared_ptr<Buffer> c) {
    if (a->dtype() != b->dtype() || a->dtype() != c->dtype()) {
        std::cerr << strings::error("buffer_ops::" + op + " error: ") << "input and output dtypes must be equal, got "
                  << strings::info(std::to_string((int)a->dtype())) << ", "
                  << strings::info(std::to_string((int)b->dtype())) << ", and "
                  << strings::info(std::to_string((int)c->dtype())) << std::endl;
        exit(-1);
    }
}

void _assert_equal_dtypes(const std::string& op, std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b) {
    if (a->dtype() != b->dtype()) {
        std::cerr << strings::error("buffer_ops::" + op + " error: ") << "input and output dtypes must be equal, got "
                  << strings::info(std::to_string((int)a->dtype())) << " and "
                  << strings::info(std::to_string((int)b->dtype())) << std::endl;
        exit(-1);
    }
}

void multiply(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("multiply", a, b, out);
    _assert_equal_dtypes("multiply", a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t a_index,
           size_t b_index, size_t out_index) {
            float output = _a->getIndex<float>(a_index) * _b->getIndex<float>(b_index);
            _out->setIndex(out_index, (void*)(&output));
        },
        a, b, out);
}

void multiply(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    for (size_t i = 0; i < out->size(); i++) {
        float output = a->getIndex<float>(i) * b;
        out->setIndex(i, (void*)(&output));
    }
}

void divide(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("divide", a, b, out);
    _assert_equal_dtypes("divide", a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t a_index,
           size_t b_index, size_t out_index) {
            if (_b->getIndex<float>(b_index) == 0.) {
                std::cerr << strings::error("buffer_ops::divide error: ") << "divide by 0 error" << std::endl;
                exit(-1);
            }

            float output = _a->getIndex<float>(a_index) / _b->getIndex<float>(b_index);
            _out->setIndex(out_index, (void*)(&output));
        },
        a, b, out);
}

void divide(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    if (b < EPSILON) {
        std::cerr << strings::error("buffer_ops::divide error: ") << "divide by 0 error" << std::endl;
        exit(-1);
    }

    for (size_t i = 0; i < out->size(); i++) {
        float output = a->getIndex<float>(i) / b;
        out->setIndex(i, (void*)(&output));
    }
}

void add(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("add", a, b, out);
    _assert_equal_dtypes("add", a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t a_index,
           size_t b_index, size_t out_index) {
            float output = _a->getIndex<float>(a_index) + _b->getIndex<float>(b_index);
            std::cout << strings::error("OUTPUT: ") << output << std::endl;
            _out->setIndex(out_index, (void*)(&output));
            std::cout << strings::error("INDEX ") << strings::info(std::to_string(out_index))
                      << strings::error(" SET TO ") << output << std::endl;
        },
        a, b, out);
}

void add(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("add", a, out);
    _assert_equal_dtypes("add", a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, float _b, std::shared_ptr<Buffer> _out, size_t index) {
            float output = _a->getIndex<float>(index) + _b;
            _out->setIndex(index, (void*)(&output));
        },
        a, b, out);
}

void subtract(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("subtract", a, b, out);
    _assert_equal_dtypes("subtract", a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t a_index,
           size_t b_index, size_t out_index) {
            float output = _a->getIndex<float>(a_index) - _b->getIndex<float>(b_index);
            _out->setIndex(out_index, (void*)(&output));
        },
        a, b, out);
}

void reciprocal(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("reciprocal", a, out);
    _assert_equal_dtypes("reciprocal", a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _out, size_t index) {
            if (_a->getIndex<float>(index) == 0.) {
                std::cerr << strings::error("buffer_ops::reciprocal error: ") << "divide by 0 error" << std::endl;
                exit(-1);
            }

            float output = 1 / _a->getIndex<float>(index);
            _out->setIndex(index, (void*)(&output));
        },
        a, out);
}

void ln(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("ln", a, out);
    _assert_equal_dtypes("ln", a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _out, size_t index) {
            // TODO: option to ignore?
            if (_a->getIndex<float>(index) <= EPSILON) {
                return;
                // std::cerr << strings::error("buffer_ops::ln error: ") << "cannot take natural log of 0" << std::endl;
                // exit(-1);
            }

            float output = std::log(_a->getIndex<float>(index));
            _out->setIndex(index, (void*)(&output));
        },
        a, out);
}

void sigmoid(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("sigmoid", a, out);
    _assert_equal_dtypes("sigmoid", a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t index) {
            float output = 1 / (1 + std::exp(-(a->getIndex<float>(index))));
            out->setIndex(index, (void*)(&output));
        },
        a, out);
}

void relu(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("relu", a, out);
    _assert_equal_dtypes("relu", a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t index) {
            float output = a->getIndex<float>(index);
            output = output > EPSILON ? output : 0;
            out->setIndex(index, (void*)(&output));
        },
        a, out);
}

void pow(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("pow", a, b, out);
    _assert_equal_dtypes("pow", a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t a_index,
           size_t b_index, size_t out_index) {
            float output = std::pow(_a->getIndex<float>(a_index), _b->getIndex<float>(b_index));
            _out->setIndex(out_index, (void*)(&output));
        },
        a, b, out);
}

void pow(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes("pow", a, out);
    _assert_equal_dtypes("pow", a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, float _b, std::shared_ptr<Buffer> _out, size_t index) {
            float output = std::pow(_a->getIndex<float>(index), _b);
            _out->setIndex(index, (void*)(&output));
        },
        a, b, out);
}

// TODO: permutations
// TODO: this can and needs to be more efficient
//
// things are currently laid out row-wise in memory
// this might change? as such this function will need to change if/when that happens
void transpose(std::shared_ptr<Buffer> a, const std::vector<int>& shape_a, std::shared_ptr<Buffer> out,
               const std::vector<int>& shape_out) {
    _assert_equal_sizes("transpose", a, out);

    size_t matrix_count = 1;
    for (int i = 0; i < shape_a.size() - 2; i++) {
        matrix_count *= shape_a[i];

        if (shape_a[i] != shape_out[i]) {
            std::cerr << strings::error("buffer_ops::transpose error: ")
                      << "incompatible input/output batch shapes, got " << strings::info(strings::vecToString(shape_a))
                      << " and " << strings::info(strings::vecToString(shape_out)) << std::endl;
            exit(-1);
        }
    }

    int a_row_count = shape_a[shape_a.size() - 2];
    int a_col_count = shape_a[shape_a.size() - 1];
    size_t matrix_size = a_row_count * a_col_count;

    for (size_t i = 0; i < matrix_count; i++) {
        for (size_t r = 0; r < a_row_count; r++) {
            for (size_t c = 0; c < a_col_count; c++) {
                size_t input_index = i * matrix_size + (r * a_col_count + c);
                size_t output_index = i * matrix_size + (c * a_row_count + r);

                float value = a->getIndex<float>(input_index);
                out->setIndex(output_index, (void*)(&value));
            }
        }
    }
}

// this is basically copy + paste from `kernel.cpp`
void matmul(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out,
            const std::vector<int>& shape_a, const std::vector<int>& shape_b, const std::vector<int>& shape_out) {
    std::cout << strings::debug("ENTER buffer_ops::matmul") << std::endl;
    int l = shape_a.size();
    int r = shape_b.size();
    int o = shape_out.size();

    size_t l_matrix_size = shape_a[l - 2] * shape_a[l - 1];
    size_t r_matrix_size = shape_b[r - 2] * shape_b[r - 1];
    size_t o_matrix_size = shape_out[o - 2] * shape_out[o - 1];

    // for batched matrix multiplication, only the last two dimensions are considered in the multiplication
    // the rest of the dimenions just served to act as groupings of matrices
    //
    // NOTE: shape verification is performed in `allocation.cpp`
    size_t matrix_count = 1;
    for (int i = 0; i < shape_a.size() - 2; i++) {
        matrix_count *= shape_a[i];
    }

    // NOTE: broadcasting isn't really tested all that well
    std::vector<int> l_batch_shape;
    std::vector<int> r_batch_shape;

    for (int i = 0; i < shape_a.size() - 2 || i < shape_b.size() - 2; i++) {
        if (i < shape_a.size() - 2) {
            l_batch_shape.push_back(shape_a[i]);
        }

        if (i < shape_b.size() - 2) {
            r_batch_shape.push_back(shape_b[i]);
        }
    }

    bool l_is_lesser = false;

    int padding_size = std::max(l_batch_shape.size(), r_batch_shape.size());
    l_batch_shape = broadcasting::padVector(l_batch_shape, padding_size);
    r_batch_shape = broadcasting::padVector(r_batch_shape, padding_size);

    for (int i = 0; i < l_batch_shape.size(); i++) {
        if (l_batch_shape[i] < r_batch_shape[i]) {
            l_is_lesser = true;
        } else {
            l_is_lesser = false;
        }
    }

    kernel::BroadcastIterator it = l_is_lesser ? kernel::BroadcastIterator(l_batch_shape, r_batch_shape)
                                               : kernel::BroadcastIterator(r_batch_shape, l_batch_shape);

    while (!it.end()) {
        auto [lesser_index, greater_index] = it.getIndices();
        size_t left_index = l_is_lesser ? lesser_index : greater_index;
        size_t right_index = l_is_lesser ? greater_index : lesser_index;
        size_t out_index = greater_index;

        for (int i = 0; i < shape_a[l - 2]; i++) {
            for (int j = 0; j < shape_b[r - 1]; j++) {
                float dot = 0;
                for (int k = 0; k < shape_b[r - 2]; k++) {
                    float avalue = a->getIndex<float>((left_index * l_matrix_size) + (i * shape_a[l - 1]) + k);
                    float bvalue = b->getIndex<float>((right_index * r_matrix_size) + (k * shape_b[r - 1]) + j);

                    dot += avalue * bvalue;
                }

                out->setIndex((out_index * o_matrix_size) + (i * shape_out[o - 2]) + j, (void*)(&dot));
            }
        }

        it.increment();
    }
    std::cout << strings::debug("EXIT buffer_ops::matmul") << std::endl;
}

float reduceSum(std::shared_ptr<Buffer> a) {
    float output = 0;
    for (size_t i = 0; i < a->size(); i++) {
        output += a->getIndex<float>(i);
    }

    return output;
}

void set(std::shared_ptr<Buffer> a, float value) {
    for (size_t i = 0; i < a->size(); i++) {
        a->setIndex(i, (void*)(&value));
    }
}

}  // namespace buffer_ops
