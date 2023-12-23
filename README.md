# Hive

My playground for tensor processing and low-level deep learning algorithms. This is a machine learning framework designed to help me understand
what makes these sorts of frameworks productive, efficient, and easy to use. It also serves as a basic inquiry into language writing and the 
possibility of a tensor programming language

This project is still heavy in development and not at all suitable for use in others. Abstractions are limited with next to no optimization.

# Examples
Hive models are defined in `.nn` files which have a very basic syntax (see the `nn/` folder for examples). For an example of training, change the `dataset_path` in `main.cpp`
and compile with `g++ ./src/*.cpp ./src/data/*.cpp main.cpp -o test -I./include -I./include/data -std=c++2a`.
