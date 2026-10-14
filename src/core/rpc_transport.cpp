#include "strata/core/rpc_transport.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <chrono>

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
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
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
bool would_block() {
    const int e = WSAGetLastError();
    return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS || e == WSAEALREADY;
}
bool interrupted() { return WSAGetLastError() == WSAEINTR; }
bool set_nonblocking(NativeSocket s) {
    u_long mode = 1;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
}
#else
using NativeSocket = int;
constexpr NativeSocket kBad = -1;
NativeSocket native(Socket s) { return (int)s; }
Socket portable(NativeSocket s) { return (Socket)s; }
bool startup(std::string&) { return true; }
std::string socket_error(const std::string& what) { return what + ": " + std::strerror(errno); }
bool would_block() { return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINPROGRESS || errno == EALREADY; }
bool interrupted() { return errno == EINTR; }
bool set_nonblocking(NativeSocket s) {
    const int flags = fcntl(s, F_GETFL, 0);
    return flags >= 0 && fcntl(s, F_SETFL, flags | O_NONBLOCK) == 0;
}
#endif

bool check_io_state(const IoDeadline& deadline, std::string& err) {
    if (deadline.cancelled != nullptr && deadline.cancelled->load(std::memory_order_relaxed)) {
        err = "RPC operation cancelled";
        return false;
    }
    if (deadline.bounded && std::chrono::steady_clock::now() >= deadline.expires) {
        err = "RPC I/O deadline exceeded";
        return false;
    }
    return true;
}

bool wait_ready(Socket socket, bool writing, const IoDeadline& deadline, std::string& err) {
    for (;;) {
        if (!check_io_state(deadline, err)) return false;
        int wait_ms = -1;
        if (deadline.bounded) {
            const auto now = std::chrono::steady_clock::now();
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline.expires - now).count();
            if (remaining <= 0) remaining = 1;
            wait_ms = (int)std::min<int64_t>(remaining, deadline.cancelled == nullptr ? INT_MAX : 100);
        } else if (deadline.cancelled != nullptr) {
            wait_ms = 100;
        }
#if defined(_WIN32)
        WSAPOLLFD fd{};
        fd.fd = native(socket);
        fd.events = writing ? POLLWRNORM : POLLRDNORM;
        const int ready = WSAPoll(&fd, 1, wait_ms);
#else
        pollfd fd{};
        fd.fd = native(socket);
        fd.events = writing ? POLLOUT : POLLIN;
        const int ready = ::poll(&fd, 1, wait_ms);
#endif
        if (ready > 0) return true; // send/recv reports EOF and socket errors with operation context
        if (ready == 0 || interrupted()) continue;
        err = socket_error("RPC socket wait failed");
        return false;
    }
}

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

IoDeadline deadline_after(std::chrono::milliseconds timeout, const std::atomic<bool>* cancelled) {
    return {std::chrono::steady_clock::now() + timeout, cancelled, true};
}

IoDeadline wait_forever() {
    return {std::chrono::steady_clock::time_point::max(), nullptr, false};
}

bool connect(const std::string& endpoint, Socket& out, std::string& err) {
    return connect(endpoint, out, deadline_after(std::chrono::seconds(5)), err);
}

bool connect(const std::string& endpoint, Socket& out, const IoDeadline& deadline, std::string& err) {
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
        if (!set_nonblocking(s)) {
#if defined(_WIN32)
            closesocket(s);
#else
            ::close(s);
#endif
            continue;
        }
        if (::connect(s, it->ai_addr, (int)it->ai_addrlen) == 0) { out = portable(s); break; }
        if (would_block() || interrupted()) {
            std::string wait_err;
            if (wait_ready(portable(s), true, deadline, wait_err)) {
                int so_error = 0;
#if defined(_WIN32)
                int so_len = sizeof(so_error);
                const int opt_result = getsockopt(s, SOL_SOCKET, SO_ERROR, (char*)&so_error, &so_len);
#else
                socklen_t so_len = sizeof(so_error);
                const int opt_result = getsockopt(s, SOL_SOCKET, SO_ERROR, &so_error, &so_len);
#endif
                if (opt_result == 0 && so_error == 0) {
                    out = portable(s);
                    break;
                }
                if (opt_result != 0) {
                    err = socket_error("could not inspect the RPC connection result");
                } else if (so_error != 0) {
#if defined(_WIN32)
                    err = "RPC connection failed (Winsock " + std::to_string(so_error) + ")";
#else
                    err = "RPC connection failed: " + std::string(std::strerror(so_error));
#endif
                }
            } else {
                err = wait_err;
            }
        }
#if defined(_WIN32)
        closesocket(s);
#else
        ::close(s);
#endif
    }
    freeaddrinfo(list);
    if (out == kInvalidSocket) {
        if (err.empty()) err = socket_error("could not connect to " + endpoint);
        else err += " while connecting to " + endpoint;
        return false;
    }
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
    if (!set_nonblocking(s)) {
        err = socket_error("could not make accepted RPC socket nonblocking");
        close(out);
        out = kInvalidSocket;
        return false;
    }
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
    return send_all(socket, data, bytes, IoDeadline{}, err);
}

bool send_all(Socket socket, const void* data, size_t bytes, const IoDeadline& deadline, std::string& err) {
    const char* p = (const char*)data;
    while (bytes) {
        if (!check_io_state(deadline, err)) return false;
        const int chunk = (int)std::min<size_t>(bytes, (size_t)INT_MAX);
#if defined(_WIN32)
        const int n = ::send(native(socket), p, chunk, 0);
#else
        const int n = (int)::send(native(socket), p, (size_t)chunk, MSG_NOSIGNAL);
#endif
        if (n < 0 && interrupted()) continue;
        if (n < 0 && would_block()) {
            if (!wait_ready(socket, true, deadline, err)) return false;
            continue;
        }
        if (n <= 0) { err = socket_error("RPC send failed"); return false; }
        p += n; bytes -= (size_t)n;
        // A nonblocking socket can make repeated immediate progress without ever entering wait_ready(). Keep the
        // caller's deadline total across the whole frame, including the final successful write.
        if (!check_io_state(deadline, err)) return false;
    }
    return true;
}

bool recv_all(Socket socket, void* data, size_t bytes, std::string& err) {
    return recv_all(socket, data, bytes, IoDeadline{}, err);
}

bool recv_all(Socket socket, void* data, size_t bytes, const IoDeadline& deadline, std::string& err) {
    char* p = (char*)data;
    while (bytes) {
        if (!check_io_state(deadline, err)) return false;
        const int chunk = (int)std::min<size_t>(bytes, (size_t)INT_MAX);
#if defined(_WIN32)
        const int n = ::recv(native(socket), p, chunk, 0);
#else
        const int n = (int)::recv(native(socket), p, (size_t)chunk, 0);
#endif
        if (n < 0 && interrupted()) continue;
        if (n < 0 && would_block()) {
            if (!wait_ready(socket, false, deadline, err)) return false;
            continue;
        }
        if (n <= 0) { err = n == 0 ? "RPC connection closed" : socket_error("RPC receive failed"); return false; }
        p += n; bytes -= (size_t)n;
        if (!check_io_state(deadline, err)) return false;
    }
    return true;
}

} // namespace strata::core::rpc
