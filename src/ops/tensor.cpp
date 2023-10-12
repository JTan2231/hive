#include <initializer_list>
#include <vector>

namespace nn_parser {

// what do i do with this ???
class TensorOp {
    std::vector<int> shape_;

   public:
    TensorOp(std::initializer_list<int> shape) {
        for (int dim : shape) {
            shape_.push_back(dim);
        }
    }
};

}  // namespace nn_parser
