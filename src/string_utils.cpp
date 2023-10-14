#include <string>

namespace strings {

bool isAlphanumeric(char c) {
    return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' ||
           c >= '0' && c <= '9' || c == '_';
}

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

}  // namespace strings
