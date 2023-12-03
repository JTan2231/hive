#ifndef CSV
#define CSV

#include <iostream>
#include <string>
#include <utility>
#include <vector>

// abstract this class with a common parent class
// for this and other readers to inherit
//
// notes:
//   - two things are default:
//     - random sampling
//     - repeated sampling (i.e. cards are put back in the deck after sampling)
//   - the entire csv file given will be read into memory
//   - a 1-dimensional vector is the *only* output shape supported -- TODO
//   - float32 is the *only* data type supported                   -- TODO
class CSVDataset {
   public:
    CSVDataset(const std::string& filepath, const std::vector<std::string>& inputs,
               const std::vector<std::string>& outputs);

    std::pair<std::vector<float>, std::vector<float>> sample();

    void printRows();

   private:
    std::vector<std::string> parseCSVRow(const std::string& row);

    void validateRow(const std::string& row);

    std::string filepath_;

    std::vector<std::string> rows_;
    std::vector<std::string> cols_;

    std::vector<int> input_cols_;
    std::vector<int> output_cols_;
};

#endif
