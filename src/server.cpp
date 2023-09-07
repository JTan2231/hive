#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

using ssize_t = int;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using SOCKET = int;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#endif

#include "server.h"

ThreadPool::ThreadPool(size_t numThreads) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queueMutex_);
                    condition_.wait(
                        lock, [this] { return stop_ || !tasks_.empty(); });

                    if (stop_ && tasks_.empty()) {
                        return;
                    }

                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        stop_ = true;
    }

    condition_.notify_all();
    for (std::thread &worker : workers_) {
        worker.join();
    }
}

template <class F>
void ThreadPool::enqueue(F &&f) {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        tasks_.emplace(std::forward<F>(f));
    }
    condition_.notify_one();
}

TcpServer::TcpServer(int port)
    : port_(port), server_fd_(INVALID_SOCKET), pool_(4) {}

TcpServer::~TcpServer() {
    if (server_fd_ != INVALID_SOCKET) {
        closeSocket(server_fd_);
    }
}

bool TcpServer::listen() {
    if (!initialize()) {
        return false;
    }

    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ == INVALID_SOCKET) {
        perror("socket failed");
        return false;
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        closeSocket(server_fd_);
        return false;
    }

    if (::listen(server_fd_, 3) < 0) {
        perror("listen");
        closeSocket(server_fd_);
        return false;
    }

    int addrlen = sizeof(address);
    SOCKET new_socket;
    std::cout << "listening on port " << port_ << std::endl;
    while (true) {
        SOCKET new_socket;
        int addrlen = sizeof(address);
        if ((new_socket = accept(server_fd_, (struct sockaddr *)&address,
                                 (socklen_t *)&addrlen)) < 0) {
            perror("accept");
            closeSocket(server_fd_);
            return false;
        }

        // Use the thread pool to handle the new connection
        pool_.enqueue([this, new_socket] { this->handleClient(new_socket); });
    }

    return true;
}

bool TcpServer::initialize() {
#if defined(_WIN32) || defined(_WIN64)
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        std::cerr << "WSAStartup failed: " << res << '\n';
        return false;
    }
#endif
    return true;
}

void TcpServer::closeSocket(SOCKET socket) {
#if defined(_WIN32) || defined(_WIN64)
    closesocket(socket);
#else
    close(socket);
#endif
}

void TcpServer::handleClient(SOCKET socket) {
    char buffer[1024] = {0};
    ssize_t valread = recv(socket, buffer, 1024, 0);
    if (valread < 0) {
        perror("recv");
    }
    std::cout << "Received: " << buffer << '\n';

    const char *response = "Hello from server";
    send(socket, response, strlen(response), 0);
    std::cout << "Response sent\n";

    closeSocket(socket);
}
