#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core::rpc {

using Socket = intptr_t;
constexpr Socket kInvalidSocket = (Socket)-1;

bool connect(const std::string& endpoint, Socket& socket, std::string& err);
bool listen(const std::string& endpoint, Socket& socket, std::string& err);
bool accept(Socket listener, Socket& socket, std::string& err);
bool set_nodelay(Socket socket, bool enabled, std::string& err);
void close(Socket socket);
bool send_all(Socket socket, const void* data, size_t bytes, std::string& err);
bool recv_all(Socket socket, void* data, size_t bytes, std::string& err);

} // namespace strata::core::rpc
