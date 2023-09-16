#ifndef MESSAGING_H
#define MESSAGING_H

#include <map>
#include <string>
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

namespace messaging {

struct Packet {
    uint16_t packetNumber;
    uint16_t totalPackets;
    std::vector<uint8_t> data;
};

enum class MessageType { INIT, HEADER, DATA, HEARTBEAT };

struct Message {
    MessageType type;
    std::map<std::string, std::string> headers;
    std::string body;
};

// TODO: break up messages that are too long

bool serializeMessage(const Message& message, std::vector<uint8_t>* buffer);
bool messageToPackets(const Message& message, std::vector<Packet> packets);

bool deserializeMessage(const std::vector<uint8_t>& buffer, Message& message);
bool deserializeMessage(const uint8_t* const buffer, size_t size,
                        Message& message);

bool sendMessage(int socket, const std::string& body, const MessageType& type);
bool sendMessage(int socket, const messaging::Message& body);

}  // namespace messaging

#endif
