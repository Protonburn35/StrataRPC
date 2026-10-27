#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <string>

namespace strata::core::rpc {

using Socket = intptr_t;
constexpr Socket kInvalidSocket = (Socket)-1;

struct IoDeadline {
    std::chrono::steady_clock::time_point expires =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    const std::atomic<bool>* cancelled = nullptr;
    bool bounded = true;
};

IoDeadline deadline_after(std::chrono::milliseconds timeout, const std::atomic<bool>* cancelled = nullptr);
IoDeadline wait_forever();

bool connect(const std::string& endpoint, Socket& socket, std::string& err);
bool connect(const std::string& endpoint, Socket& socket, const IoDeadline& deadline, std::string& err);
bool listen(const std::string& endpoint, Socket& socket, std::string& err);
bool accept(Socket listener, Socket& socket, std::string& err);
bool set_nodelay(Socket socket, bool enabled, std::string& err);
void close(Socket socket);
bool send_all(Socket socket, const void* data, size_t bytes, std::string& err);
bool send_all(Socket socket, const void* data, size_t bytes, const IoDeadline& deadline, std::string& err);
bool recv_all(Socket socket, void* data, size_t bytes, std::string& err);
bool recv_all(Socket socket, void* data, size_t bytes, const IoDeadline& deadline, std::string& err);

} // namespace strata::core::rpc
