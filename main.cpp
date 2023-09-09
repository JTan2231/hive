#include <iostream>

#include "client.h"
#include "server.h"

using namespace std;

const string address = "127.0.0.1";
constexpr int port = 8080;

int main() {
#if CLIENT
    Client client(address, port);
    bool cond = client.connectToServer();

    while (cond) {
    }
#else
    Server server(port);
    server.listen();
#endif

    return 0;
}
