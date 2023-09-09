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

class Server {
   public:
    Server(int port);
    ~Server();

    bool listen();

   private:
    bool initialize();
    void closeSocket(socket_t socket);
    void handleClient(int epoll_fd, socket_t socket);
    void handleNewClient(int epoll_fd, struct sockaddr_in& address);
    void monitorClients();

    int port_;
    socket_t server_fd_;

    std::unordered_map<int, std::chrono::system_clock::time_point>
        active_clients;
};

#endif  // TCPSERVER_H
