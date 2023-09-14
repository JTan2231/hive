#include <iostream>
#include <string>
#include <thread>

#include "client.h"
#include "messaging.h"
#include "server.h"

using namespace std;

const string address = "127.0.0.1";
constexpr int port = 8080;

void readAndSendMessage(Client& client) {
    std::string message;

    while (true) {
        std::cout << "Client: ";
        std::getline(std::cin, message);
        if (!message.empty()) {
            client.sendMessage(message, messaging::MessageType::DATA);
        }
    }
}

int main() {
#if defined(CLIENT) && CLIENT == 1
    Client client(address, port);
    bool cond = client.connectToServer();

    std::cout << "Type and hit Enter to send a message." << std::endl;
    if (cond) {
        std::thread inputThread(readAndSendMessage, std::ref(client));

        while (cond) {
        }
    }
#else
    Server server(port);
    server.listen();
#endif

    return 0;
}
