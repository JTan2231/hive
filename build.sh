#!/bin/bash

function build_project {
    local client_value=$1
    local output_name=$2
    local test_name=$3

    mkdir -p build
    cd build

    if [ -z "$test_name" ]; then
        cmake -DCLIENT_VALUE=${client_value} ..
    else
        cmake -DCLIENT_VALUE=${client_value} -DTEST_NAME=${test_name} ..
    fi
    cp compile_commands.json ..
    make
    mv Hivemind ${output_name}
    mv ${output_name} ..
    echo "./${output_name} built successfully"

    cd ..
    rm -r build
}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [node|test] [test_name]"
    exit 1
fi

if [ "$1" = "node" ]; then
    build_project 1 client
elif [ "$1" = "both" ]; then
    build_project 1 client
    build_project 0 server
elif [ "$1" = "test" ]; then
    if [ -z "$2" ]; then
        echo "Test name must be provided when using the 'test' option."
        exit 1
    fi
    build_project 1 test $2
else
    echo "Unknown option: $1"
    echo "Usage: $0 [client|server|both|test] [test_name]"
    exit 1
fi

