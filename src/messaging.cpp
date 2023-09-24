#include "messaging.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#define USE_EPOLL !_WIN64

using SOCKET = int;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)

#if !USE_EPOLL
#include <sys/select.h>
#include <sys/time.h>
#else
#include <fcntl.h>
#include <sys/epoll.h>
#endif

#include "constants.h"

namespace messaging {

bool serializeMessage(const Message& message, std::vector<uint8_t>& buffer) {
    buffer.push_back(static_cast<uint8_t>(message.type));

    uint16_t headerCount = message.headers.size();
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&headerCount),
                  reinterpret_cast<uint8_t*>(&headerCount) + sizeof(headerCount));
    for (const auto& p : message.headers) {
        const std::string& key = p.first;
        const std::string& value = p.second;

        uint16_t keyLength = key.size();
        buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&keyLength),
                      reinterpret_cast<uint8_t*>(&keyLength) + sizeof(keyLength));
        buffer.insert(buffer.end(), key.begin(), key.end());

        uint16_t valueLength = value.size();
        buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&valueLength),
                      reinterpret_cast<uint8_t*>(&valueLength) + sizeof(valueLength));
        buffer.insert(buffer.end(), value.begin(), value.end());
    }

    uint16_t bodyLength = message.body.size();
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&bodyLength),
                  reinterpret_cast<uint8_t*>(&bodyLength) + sizeof(bodyLength));
    buffer.insert(buffer.end(), message.body.begin(), message.body.end());

    return true;
}

bool serializePacket(const Packet& packet, std::vector<uint8_t>& buffer) {
    buffer.reserve(sizeof(PacketHeader) + constants::MESSAGE_BUFFER_SIZE);

    const uint8_t* header_ptr = reinterpret_cast<const uint8_t*>(&packet.header);
    buffer.insert(buffer.end(), header_ptr, header_ptr + sizeof(PacketHeader));

    buffer.insert(buffer.end(), packet.data.begin(), packet.data.end());

    return true;
}

// TODO: untested
//       server + client aren't configured to work with this either
bool messageToPackets(const Message& message, std::vector<Packet>& packets) {
    std::vector<uint8_t> buffer;
    serializeMessage(message, buffer);

    size_t totalPackets = (buffer.size() + constants::MESSAGE_BUFFER_SIZE - 1) / constants::MESSAGE_BUFFER_SIZE;

    for (size_t i = 0; i < totalPackets; ++i) {
        size_t start = i * constants::MESSAGE_BUFFER_SIZE;
        size_t end = std::min(buffer.size(), start + constants::MESSAGE_BUFFER_SIZE);

        Packet packet;
        packet.header.packet_number = i;
        packet.header.total_packets = totalPackets;
        packet.header.payload_size = end - start;
        int j = 0;
        for (auto it = buffer.begin() + start; it < buffer.begin() + end; it++, j++) {
            packet.data[j] = *it;
        }

        packets.push_back(packet);
    }

    return true;
}

bool packetsToMessage(std::vector<Packet>& packets, Message& message) {
    std::sort(packets.begin(), packets.end(),
              [](const Packet& a, const Packet& b) { return a.header.packet_number < b.header.packet_number; });

    uint32_t total_payload_size = 0;
    for (const Packet& p : packets) {
        total_payload_size += p.header.payload_size;
    }

    std::vector<uint8_t> message_buffer(total_payload_size);

    uint32_t offset = 0;
    for (const Packet& p : packets) {
        std::copy(p.data.begin(), p.data.begin() + p.header.payload_size, message_buffer.begin() + offset);
        offset += p.header.payload_size;
    }

    deserializeMessage(message_buffer, message);

    return true;
}

bool deserializePacket(const std::vector<uint8_t>& buffer, Packet& packet) {
    std::memcpy(&packet.header, buffer.data(), sizeof(PacketHeader));

    if (buffer.size() != sizeof(PacketHeader) + constants::MESSAGE_BUFFER_SIZE) {
        std::cerr << "ERROR - deserializePacket : buffer size does not match size of `struct Packet`" << std::endl;
        return false;
    }

    std::memcpy(packet.data.data(), buffer.data() + sizeof(PacketHeader), constants::MESSAGE_BUFFER_SIZE);

    return true;
}

bool deserializeMessage(const std::vector<uint8_t>& buffer, Message& message) {
    size_t offset = 0;

    if (buffer.size() < offset + 1) return false;
    message.type = static_cast<MessageType>(buffer[offset]);
    offset += 1;

    if (buffer.size() < offset + sizeof(uint16_t)) return false;
    uint16_t headerCount;
    std::memcpy(&headerCount, buffer.data() + offset, sizeof(headerCount));
    offset += sizeof(headerCount);

    for (uint16_t i = 0; i < headerCount; ++i) {
        if (buffer.size() < offset + sizeof(uint16_t)) return false;
        uint16_t keyLength;
        std::memcpy(&keyLength, buffer.data() + offset, sizeof(keyLength));
        offset += sizeof(keyLength);

        if (buffer.size() < offset + keyLength) return false;
        std::string key(buffer.data() + offset, buffer.data() + offset + keyLength);
        offset += keyLength;

        if (buffer.size() < offset + sizeof(uint16_t)) return false;
        uint16_t valueLength;
        std::memcpy(&valueLength, buffer.data() + offset, sizeof(valueLength));
        offset += sizeof(valueLength);

        if (buffer.size() < offset + valueLength) return false;
        std::string value(buffer.data() + offset, buffer.data() + offset + valueLength);
        offset += valueLength;

        message.headers[key] = value;
    }

    if (buffer.size() < offset + sizeof(uint16_t)) return false;
    uint16_t bodyLength;
    std::memcpy(&bodyLength, buffer.data() + offset, sizeof(bodyLength));
    offset += sizeof(bodyLength);

    if (buffer.size() < offset + bodyLength) return false;
    message.body.assign(buffer.data() + offset, buffer.data() + offset + bodyLength);

    return true;
}

bool deserializeMessage(const uint8_t* const buffer, size_t size, Message& message) {
    size_t offset = 0;

    // Deserialize message type
    if (size != constants::MESSAGE_BUFFER_SIZE) {
        std::cerr << "All message buffers must be " << constants::MESSAGE_BUFFER_SIZE << " bytes in size" << std::endl;

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
    for (char c : body) {
        message.body.push_back(c);
    }

    return sendMessage(socket, message);
}

bool sendMessage(int socket, const Message& message) {
    std::vector<Packet> packets;
    messageToPackets(message, packets);

    for (const auto& packet : packets) {
        std::vector<uint8_t> buffer;
        serializePacket(packet, buffer);

        if (send(socket, buffer.data(), buffer.size(), 0) < 0) {
            perror("Send failed");
            return false;
        }
    }

    return true;
}

}  // namespace messaging
