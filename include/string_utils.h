#ifndef STRING_UTILS
#define STRING_UTILS

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace strings {

bool isAlphanumeric(char c);

// misleading function name
// punctuation counts as numeric here
//
// should be isNotAlphabetic
// but that name's ugly
bool isNumeric(const std::string& s);

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

std::string randomString(size_t length);

}  // namespace strings

#endif
