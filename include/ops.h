#ifndef OPS
#define OPS

#include <unordered_set>

class Operations {
   public:
    static const std::string TENSOR;
    static const std::string MATMUL;
    static const std::string CONSTANT;
    static const std::string NORMAL;

    static bool valid(const std::string& value) { return values_.find(value) != values_.end(); }

   private:
    static const std::unordered_set<std::string> values_;
};

const std::string Operations::TENSOR = "tensor";
const std::string Operations::MATMUL = "matmul";
const std::string Operations::CONSTANT = "constant";
const std::string Operations::NORMAL = "normal";

const std::unordered_set<std::string> Operations::values_ = {Operations::TENSOR, Operations::MATMUL,
                                                             Operations::CONSTANT, Operations::NORMAL};

#endif  // define OPS