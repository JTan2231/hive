mkdir build
cd build
cmake ..
cp compile_commands.json ..
make
mv Hivemind ..
cd ..
rm -r build
