#include "data/csv.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "string_utils.h"

CSVDataset::CSVDataset(const std::string& filepath, const std::vector<std::string>& inputs,
                       const std::vector<std::string>& outputs)
    : filepath_(filepath) {
    std::ifstream file(filepath);
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            validateRow(line);
            rows_.push_back(line);
        }
        file.close();
    } else {
        std::cerr << strings::error("CSVDataset::CSVDataset error: ") << "unable to open file "
                  << strings::info(filepath) << std::endl;
    }
}

void CSVDataset::validateRow(const std::string& row) {
    if (row.size() == 0) {
        std::cerr << strings::error("CSVDataset::validateRow error: ") << "empty rows now allowed, from file "
                  << strings::info(filepath_) << std::endl;
    }

    if (cols_.size() == 0 && rows_.size() > 0) {
        std::cerr << strings::error("CSVDataset::validateRow error: ")
                  << "column names should be in the first row, from file " << strings::info(filepath_) << std::endl;
    }
}

std::vector<std::string> parseCSVRow(const std::string& row) {
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

std::vector<float> CSVDataset::sample() {
    std::vector<float> columns;

    std::string row = "";
    std::stringstream ss(row);

    std::string cell;
    bool inQuotes = false;
    while (ss.good()) {
        char c = ss.get();
        if (ss.eof()) break;

        if (inQuotes) {
            if (c == '\"' && ss.peek() == '\"') {
                ss.get();  // Skip escaped quote
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
                columns.push_back(std::stof(cell));
                cell.clear();
            } else {
                cell += c;
            }
        }
    }

    columns.push_back(std::stof(cell));

    return columns;
}

void CSVDataset::read(const std::string& filepath) {
}
