#ifndef METRICS
#define METRICS

#include <memory>

#include "buffer.h"

namespace metrics {

class Mean {
   public:
    Mean();

    float update(std::shared_ptr<Buffer> buffer);

    float value();

    void reset();

   protected:
    float sum_;
    float count_;
};

class MeanAbsoluteError : public Mean {
   public:
    MeanAbsoluteError();

    float update(std::shared_ptr<Buffer> pred, std::shared_ptr<Buffer> truth);
};

}  // namespace metrics

#endif
