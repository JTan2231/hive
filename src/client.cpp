#include "client.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include "constants.h"
#include "messaging.h"

Client::Client(const std::string& serverAddress, int serverPort)
    : serverAddress_(serverAddress), serverPort_(serverPort), sockfd_(-1) {}

bool Client::connectToServer() {
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

    std::thread(&Client::startHeartbeat, this).detach();

    return true;
}

bool Client::sendMessage(const std::string& body,
                         const messaging::MessageType& type) {
    return messaging::sendMessage(sockfd_, body, type);
}

bool Client::sendMessage(const messaging::Message& message) {
    return messaging::sendMessage(sockfd_, message);
}

bool Client::receiveMessage() {
    char buffer[1024] = {0};
    int valread = recv(sockfd_, buffer, 1024, 0);
    std::cout << "valread: " << valread << std::endl;
    if (valread < 0) {
        perror("Recv failed");
        return false;
    }

    std::cout << "Received: " << buffer << '\n';
    return true;
}

bool Client::sendMessageInPackets(
    const messaging::Message& header,
    const std::vector<messaging::Message>& packets) {
    // send header
    if (!sendMessage(header)) {
        std::cerr << "Failed to send header" << std::endl;
        return false;
    }

    // acknowledgement
    if (!receiveMessage()) {
        std::cerr << "Failed to receive acknowledgement" << std::endl;
        return false;
    }

    // Here you should also check the contents of the received message to ensure
    // it is the expected acknowledgement
    // ...

    // send packets
    for (const messaging::Message& packet : packets) {
        if (!sendMessage(packet)) {
            std::cerr << "Failed to send data packet" << std::endl;
            return false;
        }
    }

    // TODO: acknowledgement for all packets

    return true;
}

Client::~Client() {
    std::cout << "Shutting down client." << std::endl;
    closeSocket(sockfd_);
}

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto nowAsTimePoint = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowAsTimePoint), "%Y-%m-%d %H:%M:%S");

    return ss.str();
}

void Client::startHeartbeat() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        if (!sendMessage(
                constants::HEARTBEAT_PREFIX + " " + getCurrentTimestamp(),
                messaging::MessageType::HEARTBEAT)) {
            std::cerr << "Failed to send heartbeat message.\n";
            break;
        }
    }
}

void Client::closeSocket(int& socket) {
    if (socket >= 0) {
        close(socket);
        socket = -1;
    }
}
