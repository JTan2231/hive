#include "client.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

TcpClient::TcpClient(const std::string& serverAddress, int serverPort)
    : serverAddress_(serverAddress), serverPort_(serverPort), sockfd_(-1) {}

bool TcpClient::connectToServer() {
    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd_ < 0) {
        perror("Socket creation failed");
        return false;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(serverPort_);

    if (inet_pton(AF_INET, serverAddress_.c_str(), &server_addr.sin_addr) <=
        0) {
        perror("Invalid address/ Address not supported");
        closeSocket(sockfd_);
        return false;
    }

    if (connect(sockfd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) <
        0) {
        perror("Connection Failed");
        closeSocket(sockfd_);
        return false;
    }

    return true;
}

bool TcpClient::sendMessage(const std::string& message) {
    if (send(sockfd_, message.c_str(), message.length(), 0) < 0) {
        perror("Send failed");
        return false;
    }
    return true;
}

bool TcpClient::receiveMessage() {
    char buffer[1024] = {0};
    int valread = recv(sockfd_, buffer, 1024, 0);
    if (valread < 0) {
        perror("Recv failed");
        return false;
    }

    std::cout << "Received: " << buffer << '\n';
    return true;
}

TcpClient::~TcpClient() {
    closeSocket(sockfd_);
}

void TcpClient::closeSocket(int& socket) {
    if (socket >= 0) {
        close(socket);
        socket = -1;
    }
}
