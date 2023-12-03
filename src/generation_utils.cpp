#include "generation_utils.h"

#include <random>
#include <vector>

namespace generation {

void fillNormal(std::vector<float>& output) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0, 1);

    for (int i = 0; i < output.size(); i++) {
        output[i] = dis(gen);
    }
}

// [min, max)
int randomInt(int min, int max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<std::mt19937::result_type> dist(min, max - 1);

    return dist(rng);
}

}  // namespace generation
