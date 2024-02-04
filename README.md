# Hive

My playground for understanding the technical difficulties in implementing a performant tensor processing and automatic differentation framework

This project is still heavy in development--I wouldn't recommend using this as a dependency.

# Examples
Hive models are defined in `.nn` files which have a very basic syntax (see the `nn/` folder for examples). Change `main.cpp` to use whichever of the example tests you want and compile with `g++ ./src/*.cpp ./src/data/*.cpp main.cpp -o test -I./include -I./include/data -std=c++2a`.
