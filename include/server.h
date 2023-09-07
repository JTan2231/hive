#ifndef TCPSERVER_H
#define TCPSERVER_H

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
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

class ThreadPool {
   public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();

    template <class F>
    void enqueue(F &&f);

   private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queueMutex_;
    std::condition_variable condition_;
    bool stop_ = false;
};

class TcpServer {
   public:
    TcpServer(int port);
    ~TcpServer();

    bool listen();

   private:
    bool initialize();
    void closeSocket(socket_t socket);
    void handleClient(socket_t socket);

    int port_;
    socket_t server_fd_;
    ThreadPool pool_;
};

#endif  // TCPSERVER_H
