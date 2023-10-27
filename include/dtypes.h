enum class DTYPE { float32, float64 };

namespace dtypes {

size_t dtypeSize(DTYPE dtype) {
    if (dtype == DTYPE::float32) {
        return sizeof(float);
    }

    if (dtype == DTYPE::float64) {
        return sizeof(double);
    }

    return 0;
}

float* toFloat32(void* ptr) { return (float*)ptr; }

}  // namespace dtypes