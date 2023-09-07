#include <string>

class TcpClient {
   public:
    TcpClient(const std::string& serverAddress, int serverPort);
    ~TcpClient();

    bool connectToServer();
    bool sendMessage(const std::string& message);
    bool receiveMessage();

   private:
    std::string serverAddress_;
    int serverPort_;
    int sockfd_;

    void closeSocket(int& socket);
};
