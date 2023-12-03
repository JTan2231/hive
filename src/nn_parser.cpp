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
NNParser::NNParser(size_t content_size) : content_size_(content_size), cursor_(0), line_(1), buffer_("") {
}
NNParser::NNParser(std::string contents) : content_size_(contents.size()), cursor_(0), line_(1), buffer_("") {
    lines_ = split(contents, "\n");
}

NNParser::NNParser(size_t content_size, std::vector<std::string> input_variables)
    : content_size_(content_size), cursor_(0), line_(1), buffer_("") {
    for (const auto& s : input_variables) {
        registered_variables_.insert(s);
    }
}

NNParser::~NNParser() {
}

// TODO: this should return a (as of yet unmade) computational graph object
// TODO: handle variable reassignment
// TODO: variables as arguments? how are those being handled?
std::shared_ptr<Graph> NNParser::parse(const std::string& contents) {
    std::shared_ptr<Graph> graph(new Graph());

    // if we're parsing a function, register the arguments in the graph as well
    if (registered_variables_.size() > 0) {
        for (auto& p : registered_variables_) {
            // NOTE: these variables are shapeless
            graph->createVariable(p, operations::input, {});
        }
    }

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
        if (buffer_ == variable_declarator_) {
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
                registerVariableDefinition(graph, variable_name, contents, false);

                buffer_ = "";
            }
        } else if (graph->isVariable(buffer_)) {
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
            registerVariableDefinition(graph, variable_name, contents, false);
            buffer_ = "";
        } else if (buffer_ == function_declarator_) {
            // functions have 3 rules (for now):
            //   1. The only variables in the scope are those input as arguments
            //   2. There must be exactly 1 return result
            //   3. Functions may not be declared/defined in another function definition

            incrementAndAdd(contents);  // check if this is an isolated keyword or part of a function name
            if (buffer_.back() == ' ') {
                std::string function_name = registerFunctionName(contents);
                incrementCursor();

                // are we just assuming all variables are tensors ????
                std::vector<std::string> args;
                std::string arg_buffer = "";

                // this leaves the cursor on the closing parenthesis `)`
                while (inBoundsNoError() && at(contents) != ')') {
                    arg_buffer += at(contents);
                    buffer_ += at(contents);  // this probably isn't needed
                    if (at(contents) == ',' || at(contents) == '(') {
                        arg_buffer = strip(arg_buffer);

                        if (stringIsAlphanumeric(arg_buffer)) {
                            args.push_back(arg_buffer);
                            arg_buffer = "";
                        } else {
                            std::cerr << strings::error("NNParser:parser error: ")
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

                incrementCursor();

                // functions will be subgraphs pointed to by a function node in the main graph
                // on execution their subgraphs will be executed with the given inputs
                //
                // each function call is treated as its own subgraph
                // and is allocated separately

                // [function_start, function_end)
                int function_start = cursor_;
                int function_end = function_start;
                while (cursor_ < contents.size() - 3) {
                    if (contents.substr(cursor_, 3) == "end") {
                        // check if it's not part of a variable name
                        if (std::isspace(contents[cursor_ - 1]) &&
                            (std::isspace(contents[cursor_ + 3]) || cursor_ == contents.size() - 4)) {
                            function_end = cursor_;
                            break;
                        }
                    }

                    incrementCursor();
                }

                std::string function_definition = trim(contents.substr(function_start, function_end - function_start));

                // do we have to instantiate a separate parser?
                NNParser function_parser(function_definition.size(), args);
                std::shared_ptr<Graph> function_graph = function_parser.parse(function_definition);

                registered_functions_[function_name] = function_graph;

                cursor_ = function_end + 3;

                buffer_ = "";
            }
        } else {
            incrementCursor();
        }
    }

    if (DEBUG) {
        std::cout << "Finished parsing" << std::endl;
        graph->printNodeValues();
    }

    return graph;
}

bool NNParser::isAlphanumeric(char c) {
    return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_';
}

bool NNParser::stringIsAlphanumeric(const std::string& s) {
    for (char c : s) {
        if (!isAlphanumeric(c)) {
            return false;
        }
    }

    return true;
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
    inBounds();  // safety check

    if (DEBUG == 2) {
        std::cout << "cursor incremented to " << cursor_ << std::endl;
    }
}

bool NNParser::inBounds() {
    bool in = cursor_ < content_size_;
    if (!in) {
        std::cerr << strings::error("parsing error: ") << "cursor out of bounds" << std::endl;

        exit(-1);
    }

    return in;
}

bool NNParser::inBoundsNoError() {
    return cursor_ < content_size_;
}

void NNParser::showCursor(const std::string& contents) {
    std::cout << strings::info("cursor is at the end of: ")
              << contents.substr(std::max(0, (int)cursor_ - 10), std::min((int)contents.size() - 1, 11)) << std::endl;
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
        if (!isAlphanumeric(c)) {
            std::cerr << strings::error("NNParser::registerVariableName error:") << " invalid character "
                      << strings::info("`" + std::to_string(c) + "`") << " in variable name" << std::endl;

            showCursor(contents);
            exit(-1);
        }

        variable_name += at(contents);
        buffer_ += at(contents);
        incrementCursor();
    }

    if (registered_variables_.find(variable_name) != registered_variables_.end()) {
        std::cerr << strings::error("NNParser::registerVariableName:") << " variable "
                  << strings::info("`" + variable_name + "`") << " already declared" << std::endl;
        exit(-1);
    }

    if (keywords.find(variable_name) != keywords.end()) {
        std::cerr << strings::error("NNParser::registerVariableName error:")
                  << strings::info(" `" + variable_name + "`") << " is a reserved keyword" << std::endl;

        exit(-1);
    }

    registered_variables_.insert(variable_name);

    // cursor is left at the whitespace following the variable name
    // note the variable name MUST have whitespace between it and the = sign
    return trim(variable_name);
}

std::string NNParser::registerVariableDefinition(std::shared_ptr<Graph> graph, std::string variable_name,
                                                 const std::string& contents, bool is_arg) {
    // the definition MUST start with an op
    // so we start with looking for the end of the op name,
    // which is either a `;` or `(`

    std::string op_name = "";
    if (is_arg) {
        op_name = variable_name;
    } else {
        // what is this for?
        while (inBounds() && isAlphanumeric(at(contents))) {
            op_name += at(contents);
            buffer_ += at(contents);
            incrementCursor();
        }
    }

    std::cout << strings::error("registered op name ") << strings::info(op_name) << std::endl;

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
        // TODO: why do we need the semicolons and \n?
        //       I think it has something to do with nested arguments but this feels disgusting
        //
        // this loop collects the arguments for this variable definition
        while (inBoundsNoError() && at(contents) != ')' && at(contents) != ';' && at(contents) != '\n') {
            arg_buffer += at(contents);
            buffer_ += at(contents);
            if (at(contents) == ',' || at(contents) == '(') {
                arg_buffer = strip(arg_buffer);

                if (isNumeric(arg_buffer)) {
                    args.push_back(arg_buffer);
                    arg_buffer = "";
                } else if (OperationRegistry::valid(arg_buffer) ||
                           registered_functions_.find(arg_buffer) != registered_functions_.end()) {
                    // this arg is the result of an operation
                    // get the result and attach it here
                    arg_buffer = registerVariableDefinition(graph, arg_buffer, contents, true);
                    args.push_back(arg_buffer);
                    arg_buffer = "";
                } else if (registered_variables_.find(arg_buffer) != registered_variables_.end()) {
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

    std::string variable_node_name = "";
    if (registered_functions_.find(op_name) != registered_functions_.end()) {
        variable_node_name = graph->createFunctionVariable(variable_name, args, registered_functions_[op_name]);
    } else {
        variable_node_name = graph->createVariable(variable_name, op_name, args);
    }

    if (inBoundsNoError() && at(contents) == ')') {
        incrementCursor();
    }

    return variable_node_name;
}

std::string NNParser::registerFunctionName(const std::string& contents) {
    // cursor_ is at the whitespace after the`n` of function when this
    // function starts
    while (inBounds() && at(contents) == ' ') {
        incrementCursor();
    }

    // at the function name
    std::string function_name = "";
    while (inBounds() && !std::isspace(at(contents)) && at(contents) != '(') {
        char c = at(contents);

        // function names can't contain non-alphanumeric characters
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_')) {
            std::cerr << strings::error("NNParser::registerFunctionName error:") << " invalid character "
                      << strings::info("`" + (std::string() + c) + "`") << " in function name" << std::endl;

            showCursor(contents);
            exit(-1);
        }

        function_name += at(contents);
        buffer_ += at(contents);
        incrementCursor();
    }

    if (registered_functions_.find(function_name) != registered_functions_.end()) {
        std::cerr << strings::error("NNParser::registerfunctionName error:") << " function "
                  << strings::info("`" + function_name + "`") << " already declared" << std::endl;
        exit(-1);
    }

    if (keywords.find(function_name) != keywords.end()) {
        std::cerr << strings::error("NNParser::registerfunctionName error:")
                  << strings::info(" `" + function_name + "`") << " is a reserved keyword" << std::endl;

        exit(-1);
    }

    if (at(contents) != '(') {
        std::cerr << strings::error("NNParser::registerFunctionName error:")
                  << " expected ( to begin argument list for function " << strings::info(function_name) << std::endl;
        exit(-1);
    }

    // cursor is left at the `(` beginning the argument list
    return trim(function_name);
}

std::string NNParser::registerFunctionDefinition(std::string function_name, const std::string& contents) {
    return "";
}

}  // namespace nn_parser
