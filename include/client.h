#ifndef CLIENT_H
#define CLIENT_H
#include <string>
#include <vector>

#include "messaging.h"

class Client {
   public:
    Client(const std::string& serverAddress, int serverPort);
    ~Client();

    bool connectToServer();

    bool sendMessage(const std::string& message,
                     const messaging::MessageType& type);
    bool sendMessage(const messaging::Message& message);

    bool sendMessageInPackets(const messaging::Message& header,
                              const std::vector<messaging::Message>& packets);

    bool receiveMessage();

    void startHeartbeat();

   private:
    std::string serverAddress_;
    int serverPort_;
    int sockfd_;

    void closeSocket(int& socket);
};

#endif
