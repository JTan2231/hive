#include "data/csv.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "generation_utils.h"
#include "string_utils.h"

CSVDataset::CSVDataset(const std::string& filepath, const std::vector<std::string>& inputs,
                       const std::vector<std::string>& outputs)
    : filepath_(filepath) {
    std::ifstream file(filepath);
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            validateRow(line);

            if (cols_.empty()) {
                cols_ = parseCSVRow(line);
            } else {
                rows_.push_back(line);
            }
        }
        file.close();
    } else {
        std::cerr << strings::error("CSVDataset::CSVDataset error: ") << "unable to open file "
                  << strings::info(filepath) << std::endl;
    }

    for (auto& s : inputs) {
        auto it = std::find(cols_.begin(), cols_.end(), s);
        if (it == cols_.end()) {
            std::cerr << strings::error("CSVDatset::CSVDataset error: ") << "given input column " << strings::info(s)
                      << " not in parsed csv columns " << strings::info(strings::vecToString(cols_)) << std::endl;
            exit(-1);
        } else {
            input_cols_.push_back(std::distance(cols_.begin(), it));
        }
    }

    for (auto& s : outputs) {
        auto it = std::find(cols_.begin(), cols_.end(), s);
        if (it == cols_.end()) {
            std::cerr << strings::error("CSVDatset::CSVDataset error: ") << "given output column " << strings::info(s)
                      << " not in parsed csv columns " << strings::info(strings::vecToString(cols_)) << std::endl;
            exit(-1);
        } else {
            output_cols_.push_back(std::distance(cols_.begin(), it));
        }
    }
}

// this just checks to make sure there are enough values in the row to match the number of columns in the header
// also that the header is at the top
// and that the row isn't empty
void CSVDataset::validateRow(const std::string& row) {
    if (row.size() == 0) {
        std::cerr << strings::error("CSVDataset::validateRow error: ") << "empty rows now allowed, from file "
                  << strings::info(filepath_) << std::endl;
        exit(-1);
    }

    if (cols_.size() == 0 && rows_.size() > 0) {
        std::cerr << strings::error("CSVDataset::validateRow error: ")
                  << "column names should be in the first row, from file " << strings::info(filepath_) << std::endl;
        exit(-1);
    }

    if (!cols_.empty()) {
        std::vector<std::string> columns = parseCSVRow(row);
        if (columns.size() != cols_.size()) {
            std::cerr << strings::error("CSVDataset::validateRow error: ") << "invalid number of columns -- row has "
                      << strings::info(std::to_string(columns.size()) + " columns") << ", header states "
                      << strings::info(std::to_string(cols_.size()) + " columns") << std::endl;
            exit(-1);
        }
    }
}

std::vector<std::string> CSVDataset::parseCSVRow(const std::string& row) {
    std::vector<std::string> columns;
    std::stringstream ss(row);
    std::string cell;
    bool inQuotes = false;

    while (ss.good()) {
        char c = ss.get();
        if (ss.eof()) break;

        if (inQuotes) {
            if (c == '\"' && ss.peek() == '\"') {
                ss.get();
                cell += '\"';
            } else if (c == '\"') {
                inQuotes = false;
            } else {
                cell += c;
            }
        } else {
            if (c == '\"') {
                inQuotes = true;
            } else if (c == ',') {
                columns.push_back(cell);
                cell.clear();
            } else {
                cell += c;
            }
        }
    }

    columns.push_back(cell);
    return columns;
}

// TODO: please cache these values at some point
//       there's no need to reparse every time a data point is sampled
std::unordered_map<std::string, std::vector<float>> CSVDataset::sample(int batch_size) {
    std::unordered_map<std::string, std::vector<float>> values;

    for (int i = 0; i < batch_size; i++) {
        int row_index = generation::randomInt(0, (int)(rows_.size()));
        std::vector<std::string> row = parseCSVRow(rows_[row_index]);

        for (int i : input_cols_) {
            const std::string& s = row[i];
            if (strings::isNumber(s)) {
                values[cols_[i]].push_back(std::stof(s));
            } else {
                std::cerr << strings::error("CSVDataset::sample error: ") << "values must be unformatted numbers, got "
                          << strings::info(s) << std::endl;
                exit(-1);
            }
        }

        for (int i : output_cols_) {
            const std::string& s = row[i];
            if (strings::isNumber(s)) {
                values[cols_[i]].push_back(std::stof(s));
            } else {
                std::cerr << strings::error("CSVDataset::sample error: ") << "values must be unformatted numbers, got "
                          << strings::info(s) << std::endl;
                exit(-1);
            }
        }
    }

    return values;
}

void CSVDataset::printRows() {
    for (auto& row : rows_) {
        std::cout << row << std::endl;
    }

    std::cout << "columns: " << strings::vecToString(cols_) << std::endl;
}
