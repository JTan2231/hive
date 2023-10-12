#include <iostream>
#include <list>
#include <memory>

#include "dtypes.h"

struct PointerData {
    void* pointer;
    DTYPE dtype;
    size_t size;
};

class Allocator {
    std::list<PointerData> allocations_;

    void cleanup() {
        for (auto data : allocations_) {
            switch (data.dtype) {
                case DTYPE::float32:
                    free((float*)data.pointer);
                    break;
                case DTYPE::float64:
                    free((double*)data.pointer);
                    break;
                default:
                    std::cerr << "allocation::cleanup error: invalid dtype "
                              << (int)data.dtype << std::endl;
            }
        }
    }

    void* newDataBuffer(size_t size, DTYPE dtype) {
        void* data = nullptr;
        switch (dtype) {
            case DTYPE::float32:
                data = (void*)malloc(sizeof(float) * size);
                break;
            case DTYPE::float64:
                data = (void*)malloc(sizeof(double) * size);
                break;
            default:
                std::cerr << "allocation::newDataBuffer error: invalid dtype "
                          << (int)dtype << std::endl;
        }

        return data;
    }
};
