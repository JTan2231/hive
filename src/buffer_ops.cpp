#include "buffer_ops.h"

#include <memory>

#include "buffer.h"
#include "kernel.h"
#include "string_utils.h"

// weird mix of the kernel element-wise functions? this needs better organized
namespace buffer_ops {

void _assert_equal_sizes(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> c) {
    if (a->size() != b->size() || a->size() != c->size()) {
        std::cerr << strings::error("buffer_ops::multiply error: ") << "input and output sizes must be equal, got "
                  << strings::info(std::to_string(a->size())) << ", " << strings::info(std::to_string(b->size()))
                  << ", and " << strings::info(std::to_string(c->size())) << std::endl;
        exit(-1);
    }
}

void _assert_equal_sizes(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b) {
    if (a->size() != b->size()) {
        std::cerr << strings::error("buffer_ops::multiply error: ") << "input and output sizes must be equal, got "
                  << strings::info(std::to_string(a->size())) << " and " << strings::info(std::to_string(b->size()))
                  << std::endl;
        exit(-1);
    }
}

void _assert_equal_dtypes(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> c) {
    if (a->dtype() != b->dtype() || a->dtype() != c->dtype()) {
        std::cerr << strings::error("buffer_ops::multiply error: ") << "input and output dtypes must be equal, got "
                  << strings::info(std::to_string((int)a->dtype())) << ", "
                  << strings::info(std::to_string((int)b->dtype())) << ", and "
                  << strings::info(std::to_string((int)c->dtype())) << std::endl;
        exit(-1);
    }
}

void _assert_equal_dtypes(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b) {
    if (a->dtype() != b->dtype()) {
        std::cerr << strings::error("buffer_ops::multiply error: ") << "input and output dtypes must be equal, got "
                  << strings::info(std::to_string((int)a->dtype())) << " and "
                  << strings::info(std::to_string((int)b->dtype())) << std::endl;
        exit(-1);
    }
}

void multiply(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes(a, b, out);
    _assert_equal_dtypes(a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t index) {
            float output = _a->getIndex<float>(index) * _b->getIndex<float>(index);
            _out->setIndex(index, (void*)(&output));
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
    _assert_equal_sizes(a, b, out);
    _assert_equal_dtypes(a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t index) {
            if (_a->getIndex<float>(index) == 0.) {
                std::cerr << strings::error("buffer_ops::reciprocal error: ") << "divide by 0 error" << std::endl;
                exit(-1);
            }

            float output = _a->getIndex<float>(index) / _b->getIndex<float>(index);
            _out->setIndex(index, (void*)(&output));
        },
        a, b, out);
}

void add(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes(a, b, out);
    _assert_equal_dtypes(a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t index) {
            float output = _a->getIndex<float>(index) + _b->getIndex<float>(index);
            _out->setIndex(index, (void*)(&output));
        },
        a, b, out);
}

void add(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes(a, out);
    _assert_equal_dtypes(a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, float _b, std::shared_ptr<Buffer> _out, size_t index) {
            float output = _a->getIndex<float>(index) + _b;
            _out->setIndex(index, (void*)(&output));
        },
        a, b, out);
}

void reciprocal(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes(a, out);
    _assert_equal_dtypes(a, out);

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
    _assert_equal_sizes(a, out);
    _assert_equal_dtypes(a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _out, size_t index) {
            if (_a->getIndex<float>(index) == 0.) {
                std::cerr << strings::error("buffer_ops::ln error: ") << "cannot take natural log of 0" << std::endl;
                exit(-1);
            }

            float output = std::log(_a->getIndex<float>(index));
            _out->setIndex(index, (void*)(&output));
        },
        a, out);
}

void sigmoid(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes(a, out);
    _assert_equal_dtypes(a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t index) {
            float output = 1 / (1 + std::exp(-(a->getIndex<float>(index))));
            out->setIndex(index, (void*)(&output));
        },
        a, out);
}

void relu(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes(a, out);
    _assert_equal_dtypes(a, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out, size_t index) {
            float output = a->getIndex<float>(index);
            output = output > 0 ? output : 0;
            out->setIndex(index, (void*)(&output));
        },
        a, out);
}

void pow(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes(a, b, out);
    _assert_equal_dtypes(a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t index) {
            float output = std::pow(_a->getIndex<float>(index), _b->getIndex<float>(index));
            _out->setIndex(index, (void*)(&output));
        },
        a, b, out);
}

void pow(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    _assert_equal_sizes(a, out);
    _assert_equal_dtypes(a, out);

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
    _assert_equal_sizes(a, out);

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

void matmul(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out,
            const std::vector<int>& shape_a, const std::vector<int>& shape_b, const std::vector<int>& shape_out) {
    int n = shape_a.size();
    int m = shape_b.size();
    int p = shape_out.size();

    if (shape_a[n - 1] != shape_b[m - 2] || shape_a[n - 2] != shape_out[p - 2] || shape_b[m - 1] != shape_out[p - 1] ||
        n != m || n != p) {
        std::cerr << strings::error("buffer_ops::matmul error: ") << "incompatible input/output shapes, got "
                  << strings::info(strings::vecToString(shape_a)) << ", "
                  << strings::info(strings::vecToString(shape_b)) << " and "
                  << strings::info(strings::vecToString(shape_out)) << std::endl;
        exit(-1);
    }

    size_t matrix_count = 1;
    for (int i = 0; i < shape_a.size() - 2; i++) {
        matrix_count *= shape_a[i];

        if (shape_a[i] != shape_out[i]) {
            std::cerr << strings::error("buffer_ops::matmul error: ") << "incompatible input/output batch shapes, got "
                      << strings::info(strings::vecToString(shape_a)) << ", "
                      << strings::info(strings::vecToString(shape_b)) << " and "
                      << strings::info(strings::vecToString(shape_out)) << std::endl;
            exit(-1);
        }
    }

    size_t matrix_size_a = shape_a[n - 2] * shape_a[n - 1];
    size_t matrix_size_b = shape_b[m - 2] * shape_b[m - 1];
    size_t matrix_size_out = shape_out[p - 2] * shape_out[p - 1];

    for (size_t i = 0; i < matrix_count; i++) {
        for (int ar = 0; ar < shape_a[n - 2]; ar++) {
            for (int bc = 0; bc < shape_b[m - 1]; bc++) {
                float dot = 0;
                for (int ac = 0; ac < shape_a[n - 1]; ac++) {
                    // dot += a[ar][ac] * b[ac][bc]
                    size_t a_index = (i * matrix_size_a) + (ar * shape_a[n - 1]) + ac;
                    size_t b_index = (i * matrix_size_b) + (ac * shape_b[m - 1]) + bc;

                    dot += a->getIndex<float>(a_index) * b->getIndex<float>(b_index);
                }

                // out[ar][bc] = dot
                size_t out_index = (i * matrix_size_out) + (ar * shape_out[p - 1]) + bc;
                out->setIndex(out_index, (void*)(&dot));
            }
        }
    }
}

}  // namespace buffer_ops