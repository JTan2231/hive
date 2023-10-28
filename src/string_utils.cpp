#include "string_utils.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

std::string randomString(size_t length) {
    auto randchar = []() -> char {
        const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[rand() % max_index];
    };

    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);

    return str;
}

}  // namespace strings
