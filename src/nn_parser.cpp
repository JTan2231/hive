#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "./ops.cpp"

#define DEBUG 1

namespace nn_parser {

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
class NNParser {
   public:
    NNParser(size_t content_size) : content_size_(content_size), cursor_(0), line_(1), buffer_("") {}
    ~NNParser() {}

    // TODO: this should return a (as of yet unmade) computational graph object
    // TODO: handle variable reassignment
    // TODO: variables as arguments? how are those being handled?
    void parse(const std::string& contents) {
        // order of operations when looking at a file
        // we are looking for
        //   - a variable decalaration
        //   - a definition for the above variable
        while (inBounds()) {
            if (at(contents) != '\n') {
                buffer_ += at(contents);
            }

            if (at(contents) == ';') {
                if (DEBUG) {
                    std::cout << " -- finished line " << line_ << ": " << buffer_ << std::endl;
                }

                buffer_ = "";
                line_++;
            }

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
                    // TODO
                    registerVariableDefinition(contents);
                }
            }

            incrementCursor();
        }
    }

   private:
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

    std::string strip(const std::string& s) {
        std::string stripped = "";

        for (char c : s) {
            if (isAlphanumeric(c)) {
                stripped += c;
            }
        }

        return stripped;
    }

    void incrementCursor() {
        cursor_++;
        inBounds();  // safety check

        if (DEBUG == 2) {
            std::cout << "cursor incremented to " << cursor_ << std::endl;
        }
    }

    bool inBounds() {
        bool in = cursor_ < content_size_;
        if (!in) {
            std::cerr << "parsing error: cursor out of bounds" << std::endl;

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
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
                std::cerr << "parsing error: invalid character `" << c << "` in variable name" << std::endl;

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

        if (DEBUG) {
            std::cout << "registered variable: " << variable_name << std::endl;
        }

        // cursor is left at the whitespace following the variable name
        // note the variable name MUST have whitespace between it and the = sign
        return variable_name;
    }

    std::string registerVariableDefinition(const std::string& contents) {
        // the definition MUST start with an op
        // so we start with looking for the end of the op name,
        // which is either a `;` or `(`

        std::string op_name = "";
        while (inBounds() && isAlphanumeric(at(contents))) {
            op_name += at(contents);
            buffer_ += at(contents);
            incrementCursor();
        }

        if (DEBUG) {
            std::cout << "op name: " << op_name << std::endl;
        }

        std::vector<std::string> args;
        if (at(contents) == '(') {
            // register list of operator arguments
            buffer_ += at(contents);
            incrementCursor();

            // only expecting integer arguments for now
            // might change in the future
            std::string arg_buffer = "";
            while (inBounds() && at(contents) != ')') {
                arg_buffer += at(contents);
                buffer_ += at(contents);
                if (at(contents) == ',') {
                    arg_buffer = strip(arg_buffer);

                    if (DEBUG) {
                        std::cout << "registering arg: " << arg_buffer << std::endl;
                    }

                    if (isNumeric(arg_buffer)) {
                        args.push_back(arg_buffer);
                        arg_buffer = "";
                    } else if (Operations::valid(arg_buffer)) {
                        // this arg is the result of an operation
                        // get the result and attach it here
                        arg_buffer = registerVariableDefinition(contents);
                        args.push_back(arg_buffer);
                        arg_buffer = "";
                    } else if (registered_variables.find(arg_buffer) != registered_variables.end()) {
                        // ???
                        // link the variable here
                        args.push_back(arg_buffer);
                        arg_buffer = "";
                    }
                }

                // TODO: error handling for when the arg isn't an expression, variable, or value

                incrementCursor();
            }

            if (arg_buffer.size() > 0) {
                args.push_back(strip(arg_buffer));
            }
        } else {
            std::cerr << "NNParser::registerVariableDefinition error: expected "
                         "operator arguments"
                      << std::endl;
            exit(-1);
        }

        if (DEBUG) {
            std::cout << "ARGS ";
            for (std::string& s : args) {
                std::cout << s << " ";
            }

            std::cout << std::endl;

            std::cout << "Registered definition: " << op_name << "(";
            for (int i = 0; i < args.size() - 1; i++) {
                std::cout << args[i] << ", ";
            }

            if (args.size() > 1) {
                std::cout << args.back();
            } else {
                std::cout << args.front();
            }

            std::cout << ")" << std::endl;
        }

        buffer_ += at(contents);

        return op_name;
    }

    std::string buffer_;

    size_t content_size_;
    size_t cursor_;
    size_t line_;

    const std::string variable_declarator = "let";

    std::set<std::string> registered_variables;
};

}  // namespace nn_parser

int main() {
    const std::string filepath = "./nn/llm.nn";
    const std::string contents = nn_parser::readFile(filepath);

    nn_parser::NNParser parser(contents.size());
    parser.parse(contents);
}
