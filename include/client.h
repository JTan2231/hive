#include <string>

class Client {
   public:
    Client(const std::string& serverAddress, int serverPort);
    ~Client();

    bool connectToServer();
    bool sendMessage(const std::string& message);
    bool receiveMessage();

    void startHeartbeat();

   private:
    std::string serverAddress_;
    int serverPort_;
    int sockfd_;

    void closeSocket(int& socket);
};
