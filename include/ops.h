#ifndef OPS_H
#define OPS_H

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

class Node;

// Function prototype
using FuncType = void (*)(std::shared_ptr<Node>);

class OperationRegistry {
   public:
    static std::unordered_map<std::string, FuncType>& GetOperationMap() {
        static std::unordered_map<std::string, FuncType> opMap;
        return opMap;
    }

    static std::unordered_map<std::string, FuncType>& GetAllocationMap() {
        static std::unordered_map<std::string, FuncType> alMap;
        return alMap;
    }

    template <const char* str, FuncType opFunc, FuncType alFunc>
    struct AutoRegisterOperation {
        AutoRegisterOperation() {
            OperationRegistry::GetOperationMap()[str] = opFunc;
            OperationRegistry::GetAllocationMap()[str] = alFunc;
        }
    };

    static bool valid(const std::string& value) {
        return GetOperationMap().find(value) != GetOperationMap().end();
    }
};

#define REGISTER_OPERATION(name)                                                                           \
    namespace kernel {                                                                                     \
    void name(std::shared_ptr<Node>);                                                                      \
    }                                                                                                      \
    namespace allocation {                                                                                 \
    void name##Allocate(std::shared_ptr<Node>);                                                            \
    }                                                                                                      \
    const char _str_##name[] = #name;                                                                      \
    static OperationRegistry::AutoRegisterOperation<_str_##name, kernel::name, allocation::name##Allocate> \
        _reg_op_##name;                                                                                    \
    namespace operations {                                                                                 \
    const std::string name = _str_##name;                                                                  \
    }

REGISTER_OPERATION(matmul);
REGISTER_OPERATION(constant);
REGISTER_OPERATION(tensor);
REGISTER_OPERATION(normal);

#endif  // OPS_H

