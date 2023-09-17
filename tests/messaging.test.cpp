#include "messaging.h"

#include <iostream>
#include <vector>

using namespace std;

namespace messaging {

namespace test {

string boolToSuccess(bool b) {
    return b ? "PASS" : "FAIL";
}

vector<uint8_t> stringToVec(const string& s) {
    vector<uint8_t> out;
    for (char c : s) {
        out.push_back(c);
    }

    return out;
}

string vecToString(const vector<uint8_t>& v) {
    string out;
    for (char c : v) {
        out.push_back(c);
    }

    return out;
}

bool compareMessages(const Message& message, const Message& deserialized) {
    bool equivalent = true;
    if (message.body != deserialized.body) {
        cerr << "Body mismatch: got `" << vecToString(deserialized.body) << "` instead of `"
             << vecToString(message.body) << endl;

        equivalent = false;
    }

    if (message.headers != deserialized.headers) {
        cerr << "Headers mismatch: got " << endl;
        for (auto& p : deserialized.headers) {
            cerr << "  - " << p.first << " -> " << p.second << endl;
        }

        cerr << "instead of" << endl;

        for (auto& p : message.headers) {
            cerr << "  - " << p.first << " -> " << p.second << endl;
        }

        equivalent = false;
    }

    if (message.type != deserialized.type) {
        cerr << "Message type mismatch: got " << (int)deserialized.type << " instead of " << (int)message.type << endl;

        equivalent = false;
    }

    return equivalent;
}

void messageSerializationDeserializationTest() {
    messaging::Message message;
    vector<uint8_t> buffer;

    string body = "This is a test";
    string header = "This is a test header";

    message.body = stringToVec(body);
    message.type = messaging::MessageType::DATA;
    message.headers["Test-Header"] = header;

    messaging::Message deserialized;

    messaging::serializeMessage(message, buffer);
    messaging::deserializeMessage(buffer, deserialized);

    cout << "Message de/serialization test: " << boolToSuccess(compareMessages(message, deserialized)) << endl;
}

void packetSerializationDeserializationTest() {
    Packet packet;
    std::vector<uint8_t> buffer;

    packet.header.packet_number = 1;
    packet.header.total_packets = 5;
    packet.header.payload_size = constants::MESSAGE_BUFFER_SIZE;

    for (size_t i = 0; i < constants::MESSAGE_BUFFER_SIZE; ++i) {
        packet.data[i] = static_cast<uint8_t>(i % 256);
    }

    Packet deserialized;

    serializePacket(packet, buffer);
    bool successfulDeserialization = deserializePacket(buffer, deserialized);

    bool equivalent = true;

    if (!successfulDeserialization) {
        cerr << "Deserialization failed due to size mismatch." << endl;
        equivalent = false;
    }

    if (packet.header.packet_number != deserialized.header.packet_number ||
        packet.header.total_packets != deserialized.header.total_packets ||
        packet.header.payload_size != deserialized.header.payload_size) {
        cerr << "Header mismatch: got {" << endl
             << "  " << deserialized.header.packet_number << ", "
             << "  " << deserialized.header.total_packets << ", "
             << "  " << deserialized.header.payload_size << endl
             << "}" << endl
             << "instead of" << endl
             << "{" << endl
             << "  " << packet.header.packet_number << ", "
             << "  " << packet.header.total_packets << ", "
             << "  " << packet.header.payload_size << endl
             << "}" << endl;
        equivalent = false;
    }

    // Check data
    if (packet.data != deserialized.data) {
        cerr << "Data mismatch." << endl;
        equivalent = false;
    }

    cout << "Packet de/serialization test: " << boolToSuccess(equivalent) << endl;
}

void messagePacketPipelineTest() {
    messaging::Message message;
    vector<uint8_t> buffer;

    string body = "This is a test";
    for (int i = 0; i < 10; i++) {
        body += body;
    }

    string header = "This is a test header";

    message.body = stringToVec(body);
    message.type = messaging::MessageType::DATA;
    message.headers["Test-Header"] = header;

    messaging::Message deserialized;

    vector<Packet> packets;

    messageToPackets(message, packets);
    packetsToMessage(packets, deserialized);

    cout << "Message <-> packet pipeline test: " << boolToSuccess(compareMessages(message, deserialized)) << endl;
}

}  // namespace test

}  // namespace messaging

int main() {
    messaging::test::messageSerializationDeserializationTest();
    messaging::test::packetSerializationDeserializationTest();
    messaging::test::messagePacketPipelineTest();
}
