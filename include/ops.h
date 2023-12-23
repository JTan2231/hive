#ifndef OPS_H
#define OPS_H

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

class Node;

// Function prototype
using FuncType = void (*)(std::shared_ptr<Node>);

// is there a better way of doing this?
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

    static std::unordered_map<std::string, FuncType>& GetGradientMap() {
        static std::unordered_map<std::string, FuncType> gradMap;
        return gradMap;
    }

    template <const char* str, FuncType opFunc, FuncType alFunc, FuncType gradFunc>
    struct AutoRegisterOperation {
        AutoRegisterOperation() {
            OperationRegistry::GetOperationMap()[str] = opFunc;
            OperationRegistry::GetAllocationMap()[str] = alFunc;
            OperationRegistry::GetGradientMap()[str] = gradFunc;
        }
    };

    static bool valid(const std::string& value) {
        return GetOperationMap().find(value) != GetOperationMap().end();
    }
};

// three things are required for an operation registration:
//   - a function for the kernel computation
//   - a function for memory allocation
//   - a function for gradient calculation
//
// operation names are accessible through the operations namespace
// e.g. operations::tensor
#define REGISTER_OPERATION(name)                                                                           \
    namespace kernel {                                                                                     \
    void name(std::shared_ptr<Node>);                                                                      \
    }                                                                                                      \
    namespace allocation {                                                                                 \
    void name##Allocate(std::shared_ptr<Node>);                                                            \
    }                                                                                                      \
    namespace gradient {                                                                                   \
    void name##Gradient(std::shared_ptr<Node>);                                                            \
    }                                                                                                      \
    const char _str_##name[] = #name;                                                                      \
    static OperationRegistry::AutoRegisterOperation<_str_##name, kernel::name, allocation::name##Allocate, \
                                                    gradient::name##Gradient>                              \
        _reg_op_##name;                                                                                    \
    namespace operations {                                                                                 \
    const std::string name = _str_##name;                                                                  \
    }

REGISTER_OPERATION(input);
REGISTER_OPERATION(constant);
REGISTER_OPERATION(tensor);

REGISTER_OPERATION(normal);
REGISTER_OPERATION(ones);

REGISTER_OPERATION(add);
REGISTER_OPERATION(subtract);
REGISTER_OPERATION(multiply);
REGISTER_OPERATION(divide);

REGISTER_OPERATION(sqrt);
REGISTER_OPERATION(exp);
REGISTER_OPERATION(pow);

REGISTER_OPERATION(matmul);

REGISTER_OPERATION(sigmoid);
REGISTER_OPERATION(relu);

REGISTER_OPERATION(reduce_sum);

#endif  // OPS_H
