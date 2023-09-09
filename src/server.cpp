#include "server.h"

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
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

using SOCKET = int;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#endif

#include "constants.h"

void printDebug(const std::string &message) {
    std::cout << "DEBUG: " << message << std::endl;
}

Server::Server(int port) : port_(port), server_fd_(INVALID_SOCKET) {}

Server::~Server() {
    if (server_fd_ != INVALID_SOCKET) {
        closeSocket(server_fd_);
    }
}

bool Server::listen() {
    if (!initialize()) {
        return false;
    }

    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ == INVALID_SOCKET) {
        perror("socket failed");
        return false;
    }

    // Make the server socket non-blocking
    int flags = fcntl(server_fd_, F_GETFL, 0);
    if (flags == -1) {
        perror("fcntl get flags");
        return false;
    }
    if (fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK) == -1) {
        perror("fcntl set non-blocking");
        return false;
    }

    int opt = 1;
    if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        closeSocket(server_fd_);
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

    // Setting up epoll
    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1) {
        perror("epoll_create1");
        closeSocket(server_fd_);
        return false;
    }

    struct epoll_event event;
    event.events = EPOLLIN;
    event.data.fd = server_fd_;

    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd_, &event) == -1) {
        perror("epoll_ctl");
        closeSocket(server_fd_);
        close(epoll_fd);
        return false;
    }

    std::cout << "Server listening on port " << port_ << std::endl;

    // Event loop
    while (true) {
        struct epoll_event events[10];
        int num_fds = epoll_wait(epoll_fd, events, 10, -1);
        if (num_fds == -1) {
            perror("epoll_wait");
            closeSocket(server_fd_);
            close(epoll_fd);
            return false;
        }

        for (int i = 0; i < num_fds; i++) {
            if (events[i].data.fd == server_fd_) {
                // New client connecting
                handleNewClient(epoll_fd, address);
            } else {
                // Data available to read on a client socket
                handleClient(epoll_fd, events[i].data.fd);
            }
        }
    }

    return true;
}

bool Server::initialize() {
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

void Server::closeSocket(SOCKET socket) {
#if defined(_WIN32) || defined(_WIN64)
    closesocket(socket);
#else
    close(socket);
#endif
}

// check if string is a heartbeat message
bool checkHeartbeat(const std::string &message) {
    return message.substr(0, HEARTBEAT_PREFIX.size()) == HEARTBEAT_PREFIX;
}

void Server::handleClient(int epoll_fd, SOCKET socket) {
    char buffer[1024] = {0};
    ssize_t valread = recv(socket, buffer, 1024, 0);
    if (valread <= 0) {
        if (valread < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            // Temporarily out of resources or nothing to read, try again
            // later
            return;
        }

        if (valread < 0) {
            perror("recv");
        }

        // Connection closed
        if (epoll_ctl(epoll_fd, EPOLL_CTL_DEL, socket, NULL) == -1) {
            perror("epoll_ctl: del");
        }

        closeSocket(socket);
        active_clients.erase(socket);
        return;
    }

    buffer[valread] = '\0';

    std::string received_message(buffer);
    if (checkHeartbeat(received_message)) {
        active_clients[(int)socket] = std::chrono::system_clock::now();
    }

    std::cout << "Received \"" << buffer << "\" from connection " << (int)socket
              << " -- " << active_clients.size() << " total connections."
              << std::endl;

    const char *response = "Hello from server";
    send(socket, response, strlen(response), 0);
}

void Server::handleNewClient(int epoll_fd, struct sockaddr_in &address) {
    int addrlen = sizeof(address);
    SOCKET new_socket =
        accept(server_fd_, (struct sockaddr *)&address, (socklen_t *)&addrlen);
    if (new_socket < 0) {
        perror("accept");
        return;
    }

    // Make the new socket non-blocking
    int flags = fcntl(new_socket, F_GETFL, 0);
    if (flags == -1) {
        perror("fcntl");
        closeSocket(new_socket);
        return;
    }
    flags |= O_NONBLOCK;
    if (fcntl(new_socket, F_SETFL, flags) == -1) {
        perror("fcntl");
        closeSocket(new_socket);
        return;
    }

    // Add the new socket to the epoll instance
    struct epoll_event event;
    event.events = EPOLLIN;
    event.data.fd = new_socket;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, new_socket, &event) == -1) {
        perror("epoll_ctl");
        closeSocket(new_socket);
        return;
    }

    auto now = std::chrono::system_clock::now();
    active_clients[(int)new_socket] = now;

    std::cout << "New client connected on socket " << new_socket << std::endl;
}

void Server::monitorClients() {
    while (true) {
        std::this_thread::sleep_for(
            std::chrono::seconds(30));  // Adjust the interval as necessary

        auto now = std::chrono::system_clock::now();
        for (auto it = active_clients.begin(); it != active_clients.end();) {
            if (now - it->second >
                std::chrono::seconds(60)) {  // Adjust the timeout as necessary
                it = active_clients.erase(it);
            } else {
                ++it;
            }
        }
    }
}
