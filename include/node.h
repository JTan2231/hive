#ifndef TCPNODE_H
#define TCPNODE_H

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define USE_EPOLL !_WIN64

using SOCKET = int;
using socket_t = int;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)

#if !USE_EPOLL
#include <sys/select.h>
#include <sys/time.h>
#else
#include <fcntl.h>
#include <sys/epoll.h>
#endif

#include "messaging.h"

class Server {
   public:
    Server(int port);
    ~Server();

    bool listen();

   private:
    struct SocketInfo {
        int epoll_fd;
        int socket_fd;

        SocketInfo(int efd, int sfd) : epoll_fd(efd), socket_fd(sfd) {}
    };

    enum class TransmissionState {
        AWAITING_HEADER,
        AWAITING_DATA,
    };

    struct ClientSession {
        TransmissionState state = TransmissionState::AWAITING_HEADER;
        int expected_packets = 0;
        int received_packets = 0;
    };

    bool initialize();
    void closeSocket(socket_t socket);

    void logMessage(const SocketInfo& info, const messaging::Message& message);

    void handleClient(SocketInfo info);
    void handleNewClient(int epoll_fd, struct sockaddr_in& address);
    void handleDataTransmissionSession(const SocketInfo& info);
    void handleHeartbeat(const SocketInfo& info, const messaging::Message& message);

    void receiveData(const SocketInfo& info, messaging::Message& message);

    void monitorClients();

    bool sendMessage(const std::string& body, const messaging::MessageType& type);

    int port_;

    socket_t host_fd_;
    int highest_fd_;

    int epoll_fd_;

    fd_set master_set_;
    fd_set read_set_;

    std::unordered_map<int, ClientSession> client_sessions_;
    std::unordered_map<int, std::chrono::system_clock::time_point> active_clients;
};

#endif  // TCPSERVER_H
