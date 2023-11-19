#ifndef BUF_OPS
#define BUF_OPS

#include <memory>

#include "buffer.h"

namespace buffer_ops {

void _assert_equal_shapes(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> c);
void _assert_equal_shapes(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b);

void _assert_equal_dtypes(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> c);
void _assert_equal_dtypes(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b);

void multiply(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out);
void multiply(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out);

void divide(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out);

void add(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out);

void reciprocal(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out);

void ln(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out);

void pow(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out);
void pow(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out);

void transpose(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out);

}  // namespace buffer_ops

#endif
