#ifndef LOGGING
#define LOGGING

#include <fstream>
#include <string>

class Logging {
   public:
    static Logging& getInstance() {
        static Logging instance;
        return instance;
    }

    std::string dateTimestamp();
    std::string timestamp();

    static void error(const std::string msg) {
        getInstance().errorImpl(msg);
    }
    static void info(const std::string msg) {
        getInstance().infoImpl(msg);
    }
    static void debug(const std::string msg) {
        getInstance().debugImpl(msg);
    }

    void setLogDirectory(const std::string directory);

   private:
    Logging();
    ~Logging();

    std::string log_directory_ = "./logs/";
    std::ofstream log_file_;
    bool console_logging_ = true;

    // Implementation details
    void errorImpl(const std::string msg);
    void infoImpl(const std::string msg);
    void debugImpl(const std::string msg);

    // Delete copy constructor and assignment operator
    Logging(Logging const&) = delete;
    void operator=(Logging const&) = delete;
};

#define ERROR(msg) Logging::error(msg);
#define INFO(msg) Logging::info(msg);
#define DEBUG(msg) Logging::debug(msg);

#endif
