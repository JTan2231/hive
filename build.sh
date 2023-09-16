#!/bin/bash

function build_project {
    local client_value=$1
    local output_name=$2

    mkdir -p build
    cd build

    cmake -DCLIENT_VALUE=${client_value} ..
    cp compile_commands.json ..
    make
    mv Hivemind ${output_name}
    mv ${output_name} ..
    echo "./${output_name} built successfully"

    cd ..
    rm -r build
}

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [client|server|both]"
    exit 1
fi

if [ "$1" = "client" ]; then
    build_project 1 client
elif [ "$1" = "server" ]; then
    build_project 0 server
elif [ "$1" = "both" ]; then
    build_project 1 client
    build_project 0 server
else
    echo "Unknown option: $1"
    echo "Usage: $0 [client|server|both]"
    exit 1
fi

