#include <initializer_list>
#include <vector>

#include "../buffer.cpp"

namespace nn_parser {

// what do i do with this ???
class TensorOp {
    std::vector<int> shape_;

    Buffer* buffer_;

   public:
    TensorOp(std::initializer_list<int> shape) {
        for (int dim : shape) {
            shape_.push_back(dim);
        }
    }
};

}  // namespace nn_parser
