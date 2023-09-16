#ifndef TCPSERVER_H
#define TCPSERVER_H

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <winsock2.h>
typedef int socklen_t;
typedef SOCKET socket_t;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
typedef int socket_t;
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
    void handleHeartbeat(const SocketInfo& info,
                         const messaging::Message& message);

    void receiveData(const SocketInfo& info, messaging::Message& message);

    void monitorClients();

    bool sendMessage(const std::string& body,
                     const messaging::MessageType& type);

    int port_;
    socket_t server_fd_;

    std::unordered_map<int, ClientSession> client_sessions_;
    std::unordered_map<int, std::chrono::system_clock::time_point>
        active_clients;
};

#endif  // TCPSERVER_H
