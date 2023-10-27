#ifndef OPS
#define OPS

#include <string>
#include <unordered_set>

class Operations {
   public:
    static const std::string TENSOR;
    static const std::string MATMUL;
    static const std::string CONSTANT;
    static const std::string NORMAL;

    static bool valid(const std::string& value) {
        return values_.find(value) != values_.end();
    }

   private:
    static const std::unordered_set<std::string> values_;
};

#endif
