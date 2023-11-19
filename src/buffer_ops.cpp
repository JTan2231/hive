#include "buffer_ops.h"

#include <memory>

#include "buffer.h"
#include "kernel.h"
#include "string_utils.h"

// weird mix of the kernel element-wise functions? this needs better organized
namespace buffer_ops {

void _assert_equal_shapes(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> c) {
    if (a->size() != b->size() || a->size() != c->size()) {
        std::cerr << strings::error("buffer_ops::multiply error: ") << "input and output sizes must be equal, got "
                  << strings::info(std::to_string(a->size())) << ", " << strings::info(std::to_string(b->size()))
                  << ", and " << strings::info(std::to_string(c->size())) << std::endl;
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

void multiply(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_shapes(a, b, out);
    _assert_equal_dtypes(a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t index) {
            float output = _a->getIndex<float>(index) * _b->getIndex<float>(index);
            _out->setIndex(index, (void*)(&output));
        },
        a, b, out);
}

void multiplyConstant(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out) {
    for (size_t i = 0; i < out->size(); i++) {
        float output = a->getIndex<float>(i) * b;
        out->setIndex(i, (void*)(&output));
    }
}

void add(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out) {
    _assert_equal_shapes(a, b, out);
    _assert_equal_dtypes(a, b, out);

    kernel::_element_wise(
        [](std::shared_ptr<Buffer> _a, std::shared_ptr<Buffer> _b, std::shared_ptr<Buffer> _out, size_t index) {
            float output = _a->getIndex<float>(index) + _b->getIndex<float>(index);
            _out->setIndex(index, (void*)(&output));
        },
        a, b, out);
}

}  // namespace buffer_ops
