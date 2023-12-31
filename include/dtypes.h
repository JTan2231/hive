#ifndef DTYPES
#define DTYPES

#include <cstddef>

enum class DTYPE { float32, float64 };

namespace dtypes {

static size_t dtypeSize(DTYPE dtype) {
    if (dtype == DTYPE::float32) {
        return sizeof(float);
    }

    if (dtype == DTYPE::float64) {
        return sizeof(float);
    }

    return 0;
}

static float* toFloat32(void* ptr) {
    return (float*)ptr;
}

}  // namespace dtypes

#endif
