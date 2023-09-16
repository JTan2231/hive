#include "messaging.h"

#include <cstring>
#include <iostream>
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

// TODO: big messages?
//       things need broken up into packets and the like

void serializeMessage(const Message& message, std::vector<uint8_t>& buffer) {
    // Serialize message type
    buffer.push_back(static_cast<uint8_t>(message.type));

    // Serialize headers
    uint16_t headerCount = message.headers.size();
    buffer.insert(
        buffer.end(), reinterpret_cast<uint8_t*>(&headerCount),
        reinterpret_cast<uint8_t*>(&headerCount) + sizeof(headerCount));
    for (const auto& p : message.headers) {
        const std::string& key = p.first;
        const std::string& value = p.second;

        // Serialize key
        uint16_t keyLength = key.size();
        buffer.insert(
            buffer.end(), reinterpret_cast<uint8_t*>(&keyLength),
            reinterpret_cast<uint8_t*>(&keyLength) + sizeof(keyLength));
        buffer.insert(buffer.end(), key.begin(), key.end());

        // Serialize value
        uint16_t valueLength = value.size();
        buffer.insert(
            buffer.end(), reinterpret_cast<uint8_t*>(&valueLength),
            reinterpret_cast<uint8_t*>(&valueLength) + sizeof(valueLength));
        buffer.insert(buffer.end(), value.begin(), value.end());
    }

    // Serialize body
    uint16_t bodyLength = message.body.size();
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&bodyLength),
                  reinterpret_cast<uint8_t*>(&bodyLength) + sizeof(bodyLength));
    buffer.insert(buffer.end(), message.body.begin(), message.body.end());
}

// TODO: untested
//       server + client aren't configured to work with this either
bool messageToPackets(const Message& message, std::vector<Packet>& packets) {
    std::vector<uint8_t> buffer;
    // (Serialize message as you did, then check buffer size and create packets)
    // ...

    size_t totalPackets = (buffer.size() + constants::MESSAGE_BUFFER_SIZE - 1) /
                          constants::MESSAGE_BUFFER_SIZE;

    for (size_t i = 0; i < totalPackets; ++i) {
        size_t start = i * constants::MESSAGE_BUFFER_SIZE;
        size_t end =
            std::min(buffer.size(), start + constants::MESSAGE_BUFFER_SIZE);

        Packet packet;
        packet.packetNumber = i;
        packet.totalPackets = totalPackets;
        packet.data.insert(packet.data.end(), buffer.begin() + start,
                           buffer.begin() + end);

        packets.push_back(packet);
    }

    return true;
}

bool deserializeMessage(const std::vector<uint8_t>& buffer, Message& message) {
    size_t offset = 0;

    // Deserialize message type
    if (buffer.size() < offset + 1) return false;
    message.type = static_cast<MessageType>(buffer[offset]);
    offset += 1;

    // Deserialize headers
    if (buffer.size() < offset + sizeof(uint16_t)) return false;
    uint16_t headerCount;
    std::memcpy(&headerCount, buffer.data() + offset, sizeof(headerCount));
    offset += sizeof(headerCount);

    for (uint16_t i = 0; i < headerCount; ++i) {
        // Deserialize key
        if (buffer.size() < offset + sizeof(uint16_t)) return false;
        uint16_t keyLength;
        std::memcpy(&keyLength, buffer.data() + offset, sizeof(keyLength));
        offset += sizeof(keyLength);

        if (buffer.size() < offset + keyLength) return false;
        std::string key(buffer.data() + offset,
                        buffer.data() + offset + keyLength);
        offset += keyLength;

        // Deserialize value
        if (buffer.size() < offset + sizeof(uint16_t)) return false;
        uint16_t valueLength;
        std::memcpy(&valueLength, buffer.data() + offset, sizeof(valueLength));
        offset += sizeof(valueLength);

        if (buffer.size() < offset + valueLength) return false;
        std::string value(buffer.data() + offset,
                          buffer.data() + offset + valueLength);
        offset += valueLength;

        message.headers[key] = value;
    }

    // Deserialize body
    if (buffer.size() < offset + sizeof(uint16_t)) return false;
    uint16_t bodyLength;
    std::memcpy(&bodyLength, buffer.data() + offset, sizeof(bodyLength));
    offset += sizeof(bodyLength);

    if (buffer.size() < offset + bodyLength) return false;
    message.body.assign(buffer.data() + offset,
                        buffer.data() + offset + bodyLength);

    return true;
}

bool deserializeMessage(const int* const buffer, size_t size,
                        Message& message) {
    size_t offset = 0;

    // Deserialize message type
    if (size != constants::MESSAGE_BUFFER_SIZE) {
        std::cerr << "All message buffers must be "
                  << constants::MESSAGE_BUFFER_SIZE << " bytes in size"
                  << std::endl;

        return false;
    }

    message.type = static_cast<MessageType>(buffer[offset]);
    offset += 1;

    // Deserialize headers
    uint16_t headerCount;
    std::memcpy(&headerCount, buffer + offset, sizeof(headerCount));
    offset += sizeof(headerCount);

    for (uint16_t i = 0; i < headerCount; ++i) {
        // Deserialize key
        uint16_t keyLength;
        std::memcpy(&keyLength, buffer + offset, sizeof(keyLength));
        offset += sizeof(keyLength);

        std::string key(buffer + offset, buffer + offset + keyLength);
        offset += keyLength;

        // Deserialize value
        uint16_t valueLength;
        std::memcpy(&valueLength, buffer + offset, sizeof(valueLength));
        offset += sizeof(valueLength);

        std::string value(buffer + offset, buffer + offset + valueLength);
        offset += valueLength;

        message.headers[key] = value;
    }

    // Deserialize body
    uint16_t bodyLength;
    std::memcpy(&bodyLength, buffer + offset, sizeof(bodyLength));
    offset += sizeof(bodyLength);

    message.body.assign(buffer + offset, buffer + offset + bodyLength);

    return true;
}

bool sendMessage(int socket, const std::string& body, const MessageType& type) {
    Message message;
    message.type = type;
    message.body = body;

    std::vector<uint8_t> buffer;
    serializeMessage(message, buffer);

    if (buffer.size() > 0) {
        if (send(socket, buffer.data(), buffer.size(), 0) < 0) {
            perror("Send failed");
            return false;
        }
    } else {
        std::cerr << "Client::sendMessage -- Empty message given with type "
                  << (int)type << std::endl;

        return false;
    }

    return true;
}

bool sendMessage(int socket, const Message& message) {
    std::vector<uint8_t> buffer;
    serializeMessage(message, buffer);

    if (buffer.size() > 0) {
        if (send(socket, buffer.data(), buffer.size(), 0) < 0) {
            perror("Send failed");
            return false;
        }
    } else {
        std::cerr << "Client::sendMessage -- Empty message given with type "
                  << (int)message.type << std::endl;

        return false;
    }

    return true;
}

}  // namespace messaging
