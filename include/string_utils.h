#ifndef STRING_UTILS
#define STRING_UTILS

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "graph.h"

namespace strings {

bool isAlphanumeric(char c);

// misleading function name
// punctuation counts as numeric here
//
// should be isNotAlphabetic
// but that name's ugly
bool isNumeric(const std::string& s);

bool isNumber(const std::string& s);

std::string strip(const std::string& s);

template <typename T>
std::string vecToString(const std::vector<T>& vec) {
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i != vec.size() - 1) {
            oss << ", ";
        }
    }

    oss << "]";
    return oss.str();
}

template <typename T>
std::string vecToString(const std::vector<std::vector<T>>& vec) {
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vecToString(vec[i]);
        if (i != vec.size() - 1) {
            oss << ", ";
        }
    }

    oss << "]";
    return oss.str();
}

std::string randomString(size_t length);

std::string error(std::string s);
std::string info(std::string s);
std::string debug(std::string s);

void _error_node(const std::string& message, std::shared_ptr<Node> n);

}  // namespace strings

#endif
