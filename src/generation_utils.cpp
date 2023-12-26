#include "generation_utils.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "buffer.h"

namespace generation {

void fillNormal(std::shared_ptr<Buffer> buffer) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float range = std::sqrt(1 / 32.);
    std::uniform_real_distribution<float> dis(-1 * range, range);

    for (int i = 0; i < buffer->size(); i++) {
        float x = dis(gen);
        buffer->setIndex(i, (void*)(&x));
    }
}

// [min, max)
int randomInt(int min, int max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<std::mt19937::result_type> dist(min, max - 1);

    return dist(rng);
}

}  // namespace generation
