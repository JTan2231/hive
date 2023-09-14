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

enum class MessageType { INIT, DATA, HEARTBEAT };

struct Message {
    MessageType type;
    std::map<std::string, std::string> headers;
    std::string body;
};

// TODO: break up messages that are too long

std::vector<uint8_t> serializeMessage(const Message& message);
bool deserializeMessage(const std::vector<uint8_t>& buffer, Message& message);
bool sendMessage(socket_t socket, const std::string& body, MessageType type);

}  // namespace messaging

#endif
