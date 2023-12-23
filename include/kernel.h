#ifndef KERNEL
#define KERNEL

#include <functional>
#include <memory>

#include "graph.h"
#include "ops.h"

namespace kernel {

#define EPSILON 1e-6

class BroadcastIterator {
   public:
    BroadcastIterator(std::vector<int> lesser, std::vector<int> greater);

    bool increment();

    bool decrement();

    void print();

    bool end();

    std::pair<size_t, size_t> getIndices();

    std::vector<int> greater_current_;
    std::vector<int> lesser_current_;

   private:
    size_t getIndex(bool greater);

    void propagateChanges();

    void resetOutOfBounds();

    void updateEnd();

    bool end_;

    std::vector<int> greater_;
    std::vector<int> lesser_;
};

bool lesserGreater(std::vector<int> a, std::vector<int> b);

// finds and applies the proper kernel to the given node
void computeNode(std::shared_ptr<Node> node);

bool broadcastable(std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b);

void _element_wise(std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t,
                                      size_t, size_t)>
                       element_function,
                   std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> b, std::shared_ptr<Buffer> out);

void _element_wise(std::function<void(std::shared_ptr<Buffer>, std::shared_ptr<Buffer>, size_t)> element_function,
                   std::shared_ptr<Buffer> a, std::shared_ptr<Buffer> out);

void _element_wise(
    std::function<void(std::shared_ptr<Buffer>, float, std::shared_ptr<Buffer>, size_t)> element_function,
    std::shared_ptr<Buffer> a, float b, std::shared_ptr<Buffer> out);

}  // namespace kernel

#endif
