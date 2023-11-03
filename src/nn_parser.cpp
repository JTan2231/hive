#include "nn_parser.h"

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
#include "string_utils.h"

namespace nn_parser {

std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

std::string readFile(const std::string& filepath) {
    std::ifstream fileStream(filepath);
    if (!fileStream.is_open()) {
        throw std::runtime_error("Failed to open the file: " + filepath);
    }

    std::stringstream stringStream;
    stringStream << fileStream.rdbuf();
    return stringStream.str();
}

// takes as input a .nn file
// and spits out a computational graph for computing it
NNParser::NNParser(size_t content_size) : content_size_(content_size), cursor_(0), line_(1), buffer_("") {}
NNParser::NNParser(std::string contents) : content_size_(contents.size()), cursor_(0), line_(1), buffer_("") {
    lines_ = split(contents, "\n");
}

NNParser::~NNParser() {}

// TODO: this should return a (as of yet unmade) computational graph object
// TODO: handle variable reassignment
// TODO: variables as arguments? how are those being handled?
Graph NNParser::parse(const std::string& contents) {
    // order of operations when looking at a file
    // we are looking for
    //   - a variable decalaration
    //   - a definition for the above variable
    while (inBoundsNoError()) {
        if (at(contents) != '\n') {
            buffer_ += at(contents);
        }

        if (at(contents) == ';') {
            if (DEBUG) {
                std::cout << "-- finished line " << line_ << ": " << std::endl << trim(lines_[line_ - 1]) << std::endl;
            }

            buffer_ = "";
            line_++;

            incrementCursor();
            while (std::isspace(at(contents))) {
                incrementCursor();
            }

            buffer_ += at(contents);
        } else if (at(contents) == '\n') {
            if (DEBUG) {
                std::cout << "-- finished line " << line_ << ": " << std::endl << trim(lines_[line_ - 1]) << std::endl;
            }

            line_++;
        }

        buffer_ = trim(buffer_);
        if (buffer_ == variable_declarator) {
            incrementAndAdd(contents);  // check if this is a keyword or
                                        // just part of a variable name
            if (buffer_.back() == ' ') {
                std::string variable_name = registerVariableName(contents);

                // find the = sign
                while (inBounds() && at(contents) == ' ') {
                    buffer_ += at(contents);
                    incrementCursor();
                }

                if (at(contents) != '=') {
                    std::cerr << "parsing error: expected = sign" << std::endl;
                    showCursor(contents);
                    exit(-1);
                }

                buffer_ += at(contents);
                incrementCursor();

                // there's gotta be whitespace before and after the = sign
                while (inBounds() && at(contents) == ' ') {
                    buffer_ += at(contents);
                    incrementCursor();
                }

                // cursor should be left at the start of the variable
                // definition
                registerVariableDefinition(variable_name, contents, false);

                buffer_ = "";
            }
        } else if (graph.isVariable(buffer_)) {
            std::string variable_name = buffer_;
            // find the = sign
            while (inBounds() && at(contents) != '=') {
                buffer_ += at(contents);
                incrementCursor();
            }

            if (at(contents) != '=') {
                std::cerr << strings::error("parsing error:") << " expected = sign" << std::endl;
                showCursor(contents);
                exit(-1);
            }

            buffer_ += at(contents);
            incrementCursor();

            // there's gotta be whitespace before and after the = sign
            while (inBounds() && at(contents) == ' ') {
                buffer_ += at(contents);
                incrementCursor();
            }

            // cursor should be left at the start of the variable
            // definition
            registerVariableDefinition(variable_name, contents, false);
            buffer_ = "";
        } else {
            incrementCursor();
        }
    }

    if (DEBUG) {
        std::cout << "Finished parsing" << std::endl;
        graph.printNodeValues();
    }

    return graph;
}

bool NNParser::isAlphanumeric(char c) {
    return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_';
}

// misleading function name
// punctuation counts as numeric here
//
// should be isNotAlphabetic
// but that name's ugly
bool NNParser::isNumeric(const std::string& s) {
    for (char c : s) {
        if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z') {
            return false;
        }
    }

    return true;
}

std::string NNParser::strip(const std::string& s) {
    std::string stripped = "";

    for (char c : s) {
        if (isAlphanumeric(c)) {
            stripped += c;
        }
    }

    return stripped;
}

void NNParser::incrementCursor() {
    cursor_++;
    inBoundsNoError();  // safety check

    if (DEBUG == 2) {
        std::cout << "cursor incremented to " << cursor_ << std::endl;
    }
}

bool NNParser::inBounds() {
    bool in = cursor_ < content_size_;
    if (!in) {
        std::cerr << "parsing error: cursor out of bounds" << std::endl;

        exit(-1);
    }

    return in;
}

bool NNParser::inBoundsNoError() {
    return cursor_ < content_size_;
}

void NNParser::showCursor(const std::string& contents) {
    std::cout << strings::info("cursor is at the end of: ") << contents.substr(cursor_ - 10, 11) << std::endl;
}

char NNParser::at(const std::string& contents) {
    return contents[cursor_];
}

void NNParser::incrementAndAdd(const std::string& contents) {
    incrementCursor();
    buffer_ += at(contents);
}

std::string NNParser::registerVariableName(const std::string& contents) {
    // cursor_ is at the whitespace after the`t` of let when this
    // function starts
    while (inBounds() && at(contents) == ' ') {
        incrementCursor();
    }

    // at the variable name
    std::string variable_name = "";
    while (inBounds() && at(contents) != ' ') {
        char c = at(contents);

        // variable names can't contain non-alphanumeric characters
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
            std::cerr << "parsing error: invalid character `" << c << "` in variable name" << std::endl;

            showCursor(contents);
            exit(-1);
        }

        variable_name += at(contents);
        buffer_ += at(contents);
        incrementCursor();
    }

    if (registered_variables.find(variable_name) != registered_variables.end()) {
        std::cerr << "parsing error: variable `" << variable_name << "` already registered" << std::endl;
        exit(-1);
    }

    registered_variables.insert(variable_name);

    // cursor is left at the whitespace following the variable name
    // note the variable name MUST have whitespace between it and the = sign
    return trim(variable_name);
}

std::string NNParser::registerVariableDefinition(std::string variable_name, const std::string& contents, bool is_arg) {
    // the definition MUST start with an op
    // so we start with looking for the end of the op name,
    // which is either a `;` or `(`

    std::string op_name = "";
    if (is_arg) {
        op_name = variable_name;
    } else {
        while (inBounds() && isAlphanumeric(at(contents))) {
            op_name += at(contents);
            buffer_ += at(contents);
            incrementCursor();
        }
    }

    if (variable_name == "") {
        variable_name = op_name;
    }

    std::vector<std::string> args;
    if (at(contents) == '(') {
        // register list of operator arguments
        buffer_ += at(contents);
        incrementCursor();

        // only expecting integer arguments for now
        // might change in the future
        std::string arg_buffer = "";
        // this condition could probably be better generalized
        while (inBoundsNoError() && at(contents) != ')' && at(contents) != ';' && at(contents) != '\n') {
            arg_buffer += at(contents);
            buffer_ += at(contents);
            if (at(contents) == ',' || at(contents) == '(') {
                arg_buffer = strip(arg_buffer);

                if (isNumeric(arg_buffer)) {
                    args.push_back(arg_buffer);
                    arg_buffer = "";
                } else if (OperationRegistry::valid(arg_buffer)) {
                    // this arg is the result of an operation
                    // get the result and attach it here
                    arg_buffer = registerVariableDefinition(arg_buffer, contents, true);
                    args.push_back(arg_buffer);
                    arg_buffer = "";
                } else if (registered_variables.find(arg_buffer) != registered_variables.end()) {
                    // ???
                    // link the variable here
                    args.push_back(arg_buffer);
                    arg_buffer = "";
                } else {
                    std::cerr << strings::error("NNParser:registerVariableDefinition error: ")
                              << "expected expression, variable, or value " << std::endl;

                    showCursor(contents);
                    exit(-1);
                }
            }

            incrementCursor();
        }

        arg_buffer = strip(arg_buffer);
        if (arg_buffer.size() > 0) {
            args.push_back(arg_buffer);
        }
    } else if (args.size() == 0) {
        std::cerr << strings::error("NNParser::registerVariableDefinition error:") << " expected operator arguments"
                  << std::endl;

        showCursor(contents);
        exit(-1);
    }

    if (inBoundsNoError()) {
        buffer_ += at(contents);
    }

    std::string variable_node_name = graph.createVariable(variable_name, op_name, args);

    if (inBoundsNoError() && at(contents) == ')') {
        incrementCursor();
    }

    return variable_node_name;
}

}  // namespace nn_parser
