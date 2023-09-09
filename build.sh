#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [client|server]"
    exit 1
fi

mkdir build
cd build

if [ "$1" = "client" ]; then
    cmake -DCLIENT_VALUE=1 ..
    cp compile_commands.json ..
    make
    mv Hivemind client
    mv client ..
    echo "./client built successfully"
elif [ "$1" = "server" ]; then
    cmake -DCLIENT_VALUE=0 ..
    cp compile_commands.json ..
    make
    mv Hivemind server
    mv server ..
    echo "./server built successfully"
elif [ "$1" = "both" ]; then
    cmake -DCLIENT_VALUE=1 ..
    cp compile_commands.json ..
    make
    mv Hivemind client
    mv client ..
    echo "./client built successfully"

    echo "Cleaning build..."
    rm -r ./*

    cmake -DCLIENT_VALUE=0 ..
    cp compile_commands.json ..
    make
    mv Hivemind server
    mv server ..
    echo "./server built successfully"
else
    echo "Unknown option: $1"
    echo "Usage: $0 [client|server]"
    exit 1
fi

cd ..
rm -r build
