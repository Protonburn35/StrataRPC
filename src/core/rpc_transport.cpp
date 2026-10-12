#include "strata/core/rpc_transport.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace strata::core::rpc {
namespace {
#if defined(_WIN32)
using NativeSocket = SOCKET;
constexpr NativeSocket kBad = INVALID_SOCKET;
NativeSocket native(Socket s) { return (NativeSocket)s; }
Socket portable(NativeSocket s) { return (Socket)s; }
bool startup(std::string& err) {
    static const bool ok = [] { WSADATA data{}; return WSAStartup(MAKEWORD(2, 2), &data) == 0; }();
    if (!ok) err = "Winsock startup failed";
    return ok;
}
std::string socket_error(const std::string& what) { return what + " (Winsock " + std::to_string(WSAGetLastError()) + ")"; }
#else
using NativeSocket = int;
constexpr NativeSocket kBad = -1;
NativeSocket native(Socket s) { return (int)s; }
Socket portable(NativeSocket s) { return (Socket)s; }
bool startup(std::string&) { return true; }
std::string socket_error(const std::string& what) { return what + ": " + std::strerror(errno); }
#endif

bool split_endpoint(const std::string& endpoint, std::string& host, std::string& service, std::string& err) {
    if (endpoint.empty()) { err = "RPC endpoint is empty"; return false; }
    size_t colon = std::string::npos;
    if (endpoint.front() == '[') {
        const size_t end = endpoint.find(']');
        if (end == std::string::npos || end + 1 >= endpoint.size() || endpoint[end + 1] != ':') {
            err = "RPC endpoint must be host:port or [IPv6]:port"; return false;
        }
        host = endpoint.substr(1, end - 1);
        service = endpoint.substr(end + 2);
    } else {
        colon = endpoint.rfind(':');
        if (colon == std::string::npos) { err = "RPC endpoint must be host:port"; return false; }
        host = endpoint.substr(0, colon);
        service = endpoint.substr(colon + 1);
    }
    if (service.empty() || service.find_first_not_of("0123456789") != std::string::npos) {
        err = "RPC endpoint port must be a number"; return false;
    }
    const long port = std::strtol(service.c_str(), nullptr, 10);
    if (port < 1 || port > 65535) { err = "RPC endpoint port must be in 1..65535"; return false; }
    return true;
}
} // namespace

bool connect(const std::string& endpoint, Socket& out, std::string& err) {
    out = kInvalidSocket;
    std::string host, service;
    if (!split_endpoint(endpoint, host, service, err) || !startup(err)) return false;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo* list = nullptr;
    const char* name = host.empty() ? "127.0.0.1" : host.c_str();
    const int gai = getaddrinfo(name, service.c_str(), &hints, &list);
    if (gai != 0) { err = "RPC address lookup failed for " + endpoint; return false; }
    for (addrinfo* it = list; it != nullptr; it = it->ai_next) {
        NativeSocket s = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (s == kBad) continue;
        if (::connect(s, it->ai_addr, (int)it->ai_addrlen) == 0) { out = portable(s); break; }
#if defined(_WIN32)
        closesocket(s);
#else
        ::close(s);
#endif
    }
    freeaddrinfo(list);
    if (out == kInvalidSocket) { err = socket_error("could not connect to " + endpoint); return false; }
    return true;
}

bool listen(const std::string& endpoint, Socket& out, std::string& err) {
    out = kInvalidSocket;
    std::string host, service;
    if (!split_endpoint(endpoint, host, service, err) || !startup(err)) return false;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* list = nullptr;
    const char* name = host.empty() ? nullptr : host.c_str();
    const int gai = getaddrinfo(name, service.c_str(), &hints, &list);
    if (gai != 0) { err = "RPC bind-address lookup failed for " + endpoint; return false; }
    for (addrinfo* it = list; it != nullptr; it = it->ai_next) {
        NativeSocket s = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (s == kBad) continue;
        int reuse = 1;
#if defined(_WIN32)
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
#else
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif
        if (::bind(s, it->ai_addr, (int)it->ai_addrlen) == 0 && ::listen(s, 1) == 0) {
            out = portable(s); break;
        }
#if defined(_WIN32)
        closesocket(s);
#else
        ::close(s);
#endif
    }
    freeaddrinfo(list);
    if (out == kInvalidSocket) { err = socket_error("could not listen on " + endpoint); return false; }
    return true;
}

bool accept(Socket listener, Socket& out, std::string& err) {
    NativeSocket s = ::accept(native(listener), nullptr, nullptr);
    if (s == kBad) { err = socket_error("RPC accept failed"); return false; }
    out = portable(s);
    return true;
}

bool set_nodelay(Socket socket, bool enabled, std::string& err) {
    int value = enabled ? 1 : 0;
#if defined(_WIN32)
    if (setsockopt(native(socket), IPPROTO_TCP, TCP_NODELAY, (const char*)&value, sizeof(value)) != 0) {
#else
    if (setsockopt(native(socket), IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value)) != 0) {
#endif
        err = socket_error("could not set TCP_NODELAY");
        return false;
    }
    return true;
}

void close(Socket socket) {
    if (socket == kInvalidSocket) return;
#if defined(_WIN32)
    shutdown(native(socket), SD_BOTH);
    closesocket(native(socket));
#else
    shutdown(native(socket), SHUT_RDWR);
    ::close(native(socket));
#endif
}

bool send_all(Socket socket, const void* data, size_t bytes, std::string& err) {
    const char* p = (const char*)data;
    while (bytes) {
        const int chunk = (int)std::min<size_t>(bytes, (size_t)INT_MAX);
#if defined(_WIN32)
        const int n = ::send(native(socket), p, chunk, 0);
#else
        const int n = (int)::send(native(socket), p, (size_t)chunk, MSG_NOSIGNAL);
#endif
        if (n <= 0) { err = socket_error("RPC send failed"); return false; }
        p += n; bytes -= (size_t)n;
    }
    return true;
}

bool recv_all(Socket socket, void* data, size_t bytes, std::string& err) {
    char* p = (char*)data;
    while (bytes) {
        const int chunk = (int)std::min<size_t>(bytes, (size_t)INT_MAX);
#if defined(_WIN32)
        const int n = ::recv(native(socket), p, chunk, 0);
#else
        const int n = (int)::recv(native(socket), p, (size_t)chunk, 0);
#endif
        if (n <= 0) { err = n == 0 ? "RPC connection closed" : socket_error("RPC receive failed"); return false; }
        p += n; bytes -= (size_t)n;
    }
    return true;
}

} // namespace strata::core::rpc
