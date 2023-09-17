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

#include "constants.h"

namespace messaging {

struct PacketHeader {
    uint16_t packet_number = 0;
    uint16_t total_packets = 0;

    uint32_t payload_size = 0;
};

struct Packet {
    PacketHeader header;
    std::array<uint8_t, constants::MESSAGE_BUFFER_SIZE> data;
};

enum class MessageType { INIT, HEADER, DATA, HEARTBEAT };

struct Message {
    MessageType type;
    std::map<std::string, std::string> headers;
    std::vector<uint8_t> body;
};

// TODO: break up messages that are too long

bool serializeMessage(const Message& message, std::vector<uint8_t>& buffer);

bool messageToPackets(const Message& message, std::vector<Packet>& packets);
bool packetsToMessage(std::vector<Packet>& packets, Message& message);

bool serializePacket(const Packet& packet, std::vector<uint8_t>& buffer);
bool deserializePacket(const std::vector<uint8_t>& buffer, Packet& packet);

bool deserializeMessage(const std::vector<uint8_t>& buffer, Message& message);
bool deserializeMessage(const uint8_t* const buffer, size_t size, Message& message);

bool sendMessage(int socket, const std::string& body, const MessageType& type);
bool sendMessage(int socket, const messaging::Message& body);

}  // namespace messaging

#endif
