#ifndef NN_PARSER
#define NN_PARSER

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <locale>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "graph.h"
#include "ops.h"

// this should probably be moved
#define DEBUG 0

namespace nn_parser {

static inline void _ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}

static inline void _rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

static inline void _trim(std::string& s) {
    _rtrim(s);
    _ltrim(s);
}

static inline std::string ltrim(std::string s) {
    _ltrim(s);

    return s;
}

static inline std::string rtrim(std::string s) {
    _rtrim(s);

    return s;
}

static inline std::string trim(std::string s) {
    _trim(s);

    return s;
}

std::vector<std::string> split(const std::string& s, const std::string& delimiter);

std::string readFile(const std::string& filepath);

// takes as input a .nn file
// and spits out a computational graph for computing it
class NNParser {
   public:
    NNParser(size_t content_size);
    NNParser(std::string contents);

    ~NNParser();

    // TODO: this should return a (as of yet unmade) computational graph object
    // TODO: handle variable reassignment
    // TODO: variables as arguments? how are those being handled?
    Graph parse(const std::string& contents);

   private:
    bool isAlphanumeric(char c);

    // misleading function name
    // punctuation counts as numeric here
    //
    // should be isNotAlphabetic
    // but that name's ugly
    bool isNumeric(const std::string& s);

    std::string strip(const std::string& s);

    void incrementCursor();

    bool inBounds();

    bool inBoundsNoError();

    char at(const std::string& contents);

    void incrementAndAdd(const std::string& contents);

    std::string registerVariableName(const std::string& contents);

    std::string registerVariableDefinition(std::string variable_name, const std::string& contents, bool is_arg);

    std::string buffer_;

    size_t content_size_;
    size_t cursor_;
    size_t line_;

    // for debugging
    std::vector<std::string> lines_;

    const std::string variable_declarator = "let";

    std::set<std::string> registered_variables;

    Graph graph;
};

}  // namespace nn_parser

#endif