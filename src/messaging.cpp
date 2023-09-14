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

namespace messaging {

std::vector<uint8_t> serializeMessage(const Message& message) {
    std::vector<uint8_t> buffer;

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

    return buffer;
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

bool sendMessage(socket_t socket, const std::string& body,
                 const MessageType& type) {
    Message message;
    message.type = type;
    message.body = body;

    std::vector<uint8_t> buffer = messaging::serializeMessage(message);

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

}  // namespace messaging
