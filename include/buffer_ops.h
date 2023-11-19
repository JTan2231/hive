#ifndef BUF_OPS
#define BUF_OPS

#include <memory>

#include "buffer.h"

namespace buffer_ops {

void multiply(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out);
void multiplyConstant(std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out);
void add(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out);

}  // namespace buffer_ops

#endif
