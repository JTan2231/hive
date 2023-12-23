#ifndef GENERATION
#define GENERATION

#include <memory>
#include <vector>

#include "buffer.h"

namespace generation {

void fillNormal(std::shared_ptr<Buffer> buffer);

int randomInt(int min, int max);

}  // namespace generation

#endif
