#include <string>

namespace strings {

bool isAlphanumeric(char c) { return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_'; }

// misleading function name
// punctuation counts as numeric here
//
// should be isNotAlphabetic
// but that name's ugly
bool isNumeric(const std::string& s) {
    for (char c : s) {
        if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z') {
            return false;
        }
    }

    return true;
}

std::string strip(const std::string& s) {
    std::string stripped = "";

    for (char c : s) {
        if (isAlphanumeric(c)) {
            stripped += c;
        }
    }

    return stripped;
}

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

}  // namespace strings
