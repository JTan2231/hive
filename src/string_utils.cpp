#include "string_utils.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "graph.h"

namespace strings {

bool isAlphanumeric(char c) {
    return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_';
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

// this actually checks if the string is a valid number
// does NOT take formatted numbers e.g. 1,234,567 will return false
bool isNumber(const std::string& s) {
    bool decimal = false;
    bool negative = false;

    for (char c : s) {
        if (c == '.') {
            if (decimal) {
                return false;
            } else {
                decimal = true;
            }
        } else if (c == '-') {
            if (negative) {
                return false;
            } else {
                negative = true;
            }
        } else if (c < '0' || c > '9') {
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

std::string error(std::string s) {
    return "\033[31m" + s + "\033[0m";
}

std::string info(std::string s) {
    return "\033[34m" + s + "\033[0m";
}

std::string debug(std::string s) {
    return "\033[33m" + s + "\033[0m";
}

void _error_node(const std::string& message, std::shared_ptr<Node> n) {
    std::cout << error(message) << std::endl;
    n->printNode();
    std::cout << error("END " + message) << std::endl;
}

}  // namespace strings
