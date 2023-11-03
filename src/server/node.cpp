#include "node.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#define USE_EPOLL !_WIN64

using SOCKET = int;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)

#if !USE_EPOLL
#include <sys/select.h>
#include <sys/time.h>
#else
#include <fcntl.h>
#include <sys/epoll.h>
#endif

#include "constants.h"
#include "messaging.h"

void printDebug(const std::string &message) { std::cout << "DEBUG: " << message << std::endl; }

void parseInitMessage(const std::string &initMessage, std::string &username, std::string &ip) {
    std::size_t usernamePos = initMessage.find("USERNAME:");
    std::size_t ipPos = initMessage.find(";IP:");
    std::size_t endPos = initMessage.find(";", ipPos + 1);

    if (usernamePos != std::string::npos && ipPos != std::string::npos) {
        username = initMessage.substr(usernamePos + 9, ipPos - usernamePos - 9);
        ip = initMessage.substr(ipPos + 4, endPos - ipPos - 4);
    } else {
        std::cerr << "Invalid init message format" << std::endl;
    }
}

Server::Server(int port) : port_(port), host_fd_(INVALID_SOCKET) {}

Server::~Server() {
    if (host_fd_ != INVALID_SOCKET) {
        closeSocket(host_fd_);
    }
}

bool Server::listen() {
    if (!initialize()) {
        return false;
    }

    host_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (host_fd_ == INVALID_SOCKET) {
        perror("socket failed");
        return false;
    }

#if USE_EPOLL
    // Make the server socket non-blocking for epoll
    int flags = fcntl(host_fd_, F_GETFL, 0);
    if (flags == -1) {
        perror("fcntl get flags");
        return false;
    }
    if (fcntl(host_fd_, F_SETFL, flags | O_NONBLOCK) == -1) {
        perror("fcntl set non-blocking");
        return false;
    }
#endif

    int opt = 1;
    if (setsockopt(host_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        closeSocket(host_fd_);
        return false;
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(host_fd_, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        closeSocket(host_fd_);
        return false;
    }

    if (::listen(host_fd_, 3) < 0) {
        perror("listen");
        closeSocket(host_fd_);
        return false;
    }

    std::cout << "Server listening on port " << port_ << std::endl;

#if USE_EPOLL
    // Setting up epoll
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ == -1) {
        perror("epoll_create1");
        closeSocket(host_fd_);
        return false;
    }

    struct epoll_event event;
    event.events = EPOLLIN;
    event.data.fd = host_fd_;

    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, host_fd_, &event) == -1) {
        perror("epoll_ctl");
        closeSocket(host_fd_);
        close(epoll_fd_);
        return false;
    }

    // Event loop for epoll
    while (true) {
        struct epoll_event events[10];
        int num_fds = epoll_wait(epoll_fd_, events, 10, -1);
        if (num_fds == -1) {
            perror("epoll_wait");
            closeSocket(host_fd_);
            close(epoll_fd_);
            return false;
        }

        for (int i = 0; i < num_fds; i++) {
            if (events[i].data.fd == host_fd_) {
                // New client connecting
                handleNewClient(epoll_fd_, address);
            } else {
                // Data available to read on a client socket
                handleClient(SocketInfo(epoll_fd_, events[i].data.fd));
            }
        }
    }
#else
    // Using select
    FD_ZERO(&master_set_);
    FD_SET(host_fd_, &master_set_);

    highest_fd_ = host_fd_;

    while (true) {
        read_set_ = master_set_;
        if (select(highest_fd_ + 1, &read_set_, NULL, NULL, NULL) == -1) {
            perror("select");
            closeSocket(host_fd_);
            return false;
        }

        for (int i = 0; i <= highest_fd_; i++) {
            if (FD_ISSET(i, &read_set_)) {
                if (i == host_fd_) {
                    // TODO
                    // handleNewClientWithSelect(&master_set_, &highest_fd_, address);
                } else {
                    // TODO
                    // handleClient(SocketInfo(i));
                    FD_CLR(i, &master_set_);
                }
            }
        }
    }
#endif

    return true;
}

bool Server::initialize() { return true; }

void Server::closeSocket(SOCKET socket) { close(socket); }

// receive some data using `recv`,
// read into `message`
void Server::receiveData(const SocketInfo &info, messaging::Message &message) {
    int socket = info.socket_fd;

    uint8_t buffer[1024] = {0};
    ssize_t valread = recv(socket, buffer, 1024, 0);
    if (valread <= 0) {
#if USE_EPOLL
        int epoll_fd_ = info.epoll_fd_;

        if (valread < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            // Temporarily out of resources or nothing to read, try again
            // later
            return;
        }

        // Connection closed
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, socket, NULL) == -1) {
            perror("epoll_ctl: del");
        }
#else
// For select: If valread <= 0, the connection has been closed.
// Nothing specific to do for select, as the socket will be removed
// from the fd_set in the main loop after this function exits.
#endif

        if (valread < 0) {
            perror("recv");
        }

        closeSocket(socket);
        active_clients.erase(socket);
        return;
    }

    buffer[valread] = '\0';

    messaging::deserializeMessage(static_cast<const uint8_t *const>(buffer), constants::MESSAGE_BUFFER_SIZE, message);
}

void Server::logMessage(const SocketInfo &info, const messaging::Message &message) {
    std::string body;
    for (char c : message.body) {
        body.push_back(c);
    }

    std::cout << "Received \"" << body << "\" from connection " << (int)info.socket_fd << " -- "
              << active_clients.size() << " total connections." << std::endl;
}

void Server::handleHeartbeat(const SocketInfo &info, const messaging::Message &message) {
    active_clients[info.socket_fd] = std::chrono::system_clock::now();
    logMessage(info, message);
}

void Server::handleDataTransmissionSession(const SocketInfo &info) {
    ClientSession &session = client_sessions_[info.socket_fd];

    if (session.state == TransmissionState::AWAITING_HEADER) {
    }
}

void Server::handleClient(const SocketInfo info) {
    messaging::Message message;

    receiveData(info, message);

    if (message.type == messaging::MessageType::HEARTBEAT) {
        handleHeartbeat(info, message);
    } else if (message.type == messaging::MessageType::HEADER || message.type == messaging::MessageType::DATA) {
        handleDataTransmissionSession(info);
    } else {
        std::cerr << "Unknown message type sent -- discarding." << std::endl;
    }
}

void Server::handleNewClient(struct sockaddr_in &address) {
    int addrlen = sizeof(address);
    SOCKET new_socket = accept(host_fd_, (struct sockaddr *)&address, (socklen_t *)&addrlen);
    if (new_socket < 0) {
        perror("accept");
        return;
    }

#if USE_EPOLL
    // Make the new socket non-blocking for epoll
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
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, new_socket, &event) == -1) {
        perror("epoll_ctl");
        closeSocket(new_socket);
        return;
    }
#else
    // For select: Add the socket to the master fd_set
    FD_SET(new_socket, &master_set_);
    // Update the highest file descriptor if necessary
    if (new_socket > highest_fd_) {
        highest_fd_ = new_socket;
    }
#endif

    auto now = std::chrono::system_clock::now();
    active_clients[(int)new_socket] = now;

    std::cout << "New client connected on socket " << new_socket << std::endl;
}

bool Server::sendMessage(const std::string &body, const messaging::MessageType &type) {
    return messaging::sendMessage(host_fd_, body, type);
}

void Server::monitorClients() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(30));  // Adjust the interval as necessary

        auto now = std::chrono::system_clock::now();
        for (auto it = active_clients.begin(); it != active_clients.end();) {
            if (now - it->second > std::chrono::seconds(60)) {  // Adjust the timeout as necessary
                it = active_clients.erase(it);
            } else {
                ++it;
            }
        }
    }
}
