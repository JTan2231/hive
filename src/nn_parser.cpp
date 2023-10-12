#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace nn_parser {

// takes as input a .nn file
// and spits out a computational graph for computing it
class NNParser {
   public:
    NNParser(size_t content_size)
        : content_size_(content_size), cursor_(0), buffer_("") {}
    ~NNParser() {}

    // TODO: this should return a (as of yet unmade) computational graph object
    void parse(const std::string& contents) {
        // order of operations when looking at a file
        // we are looking for
        //   - a variable decalaration
        //   - a definition for the above variable
        while (inBounds()) {
            buffer_ += at(contents);
            if (buffer_ == variable_declarator) {
                incrementAndAdd(contents);  // check if this is a keyword or
                                            // just part of a variable name
                if (buffer_.back() == ' ') {
                    std::string variable_name = registerVariableName(contents);

                    // find the = sign
                    while (inBounds() && at(contents) == ' ') {
                        incrementCursor();
                    }

                    if (at(contents) != '=') {
                        std::cerr << "parsing error: expected = sign"
                                  << std::endl;
                    }

                    incrementCursor();

                    // there's gotta be whitespace before and after the = sign
                    while (inBounds() && at(contents) == ' ') {
                        incrementCursor();
                    }

                    // cursor should be left at the start of the variable
                    // definition
                    // TODO
                    registerVariableDefinition(variable_name, contents);
                }
            }
        }
    }

   private:
    bool isAlphanumeric(char c) {
        return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' ||
               c >= '0' && c <= '9' || c == '_';
    }

    void incrementCursor() {
        cursor_++;
        inBounds();  // safety check
    }

    bool inBounds() {
        bool in = cursor_ < content_size_;
        if (!in) {
            std::cerr << "parsing error: cursor out of bounds";

            exit(-1);
        }

        return in;
    }

    char at(const std::string& contents) { return contents[cursor_]; }

    void incrementAndAdd(const std::string& contents) {
        incrementCursor();
        buffer_ += at(contents);
    }

    std::string registerVariableName(const std::string& contents) {
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
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                  (c >= 'A' && c <= 'Z'))) {
                std::cerr << "parsing error: invalid character `" << c
                          << "` in variable name" << std::endl;

                exit(-1);
            }

            variable_name += at(contents);
            incrementCursor();
        }

        if (registered_variables.find(variable_name) !=
            registered_variables.end()) {
            std::cerr << "parsing error: variable `" << variable_name
                      << "` already registered" << std::endl;
            exit(-1);
        }

        registered_variables.insert(variable_name);

        // cursor is left at the whitespace following the variable name
        // note the variable name MUST have whitespace between it and the = sign
        return variable_name;
    }

    void registerVariableDefinition(const std::string& variable_name,
                                    const std::string& contents) {
        // the definition MUST start with an op
        // so we start with looking for the end of the op name,
        // which is either a `;` or `(`

        std::string op_name = "";
        while (inBounds() && isAlphanumeric(at(contents))) {
            op_name += at(contents);
            incrementCursor();
        }

        if (at(contents) == '(') {
            // register list of operator arguments
            incrementCursor();

            // only expecting integer arguments for now
            // might change in the future
            std::vector<int> args;
            std::string arg_buffer = "";
            while (inBounds() && at(contents) != ')') {
                if (at(contents) == ',') {
                    args.push_back(std::stoi(arg_buffer));
                    arg_buffer = "";
                } else if (at(contents) >= '0' && at(contents) <= '9') {
                    arg_buffer += at(contents);
                } else if (at(contents) != ' ') {
                    std::cerr << "NNParser::registerVariableDefinition error: "
                                 "expected a number, `)`, or `,`"
                              << std::endl;
                }
            }

            if (arg_buffer.size() > 0) {
                args.push_back(std::stoi(arg_buffer));
            }
        } else if (at(contents) == ';') {
            // register operator without arguments
            // should this even exist?
        } else {
            std::cerr << "NNParser::registerVariableDefinition error: expected "
                         "operator arguments or `;`"
                      << std::endl;
            exit(-1);
        }
    }

    std::string buffer_;

    size_t content_size_;
    size_t cursor_;

    const std::string variable_declarator = "let";

    std::set<std::string> registered_operations = {
        "add",    "subtract", "divide", "multiply",
        "matmul", "concat",   "matrix", "softmax",
    };

    std::set<std::string> registered_variables;
};

}  // namespace nn_parser
