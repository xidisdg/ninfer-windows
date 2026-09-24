#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {

#ifdef _WIN32
namespace {

// Windows positional read: SetFilePointerEx + synchronous ReadFile on the plain
// handle. Direct reads use a FILE_FLAG_NO_BUFFERING handle with the same
// 4096-byte alignment contract as POSIX O_DIRECT (the fork's established port
// of the artifact I/O layer; upstream ships POSIX-only).
[[noreturn]] void fail(const std::filesystem::path& path, const char* operation, DWORD error) {
    throw ArtifactError(path.string() + ": " + operation + ": Win32 error " +
                        std::to_string(static_cast<unsigned long>(error)));
}

std::uint64_t read_at(const std::filesystem::path& path, HANDLE handle, std::uint64_t offset,
                      const std::span<std::byte>& destination) {
    LARGE_INTEGER position{};
    position.QuadPart = static_cast<LONGLONG>(offset);
    if (!::SetFilePointerEx(handle, position, nullptr, FILE_BEGIN)) {
        fail(path, "set file pointer", ::GetLastError());
    }
    DWORD read = 0;
    if (!::ReadFile(handle, destination.data(), static_cast<DWORD>(destination.size()), &read,
                    nullptr)) {
        fail(path, "read", ::GetLastError());
    }
    return static_cast<std::uint64_t>(read);
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    // Share write access like POSIX readers: a reader must not exclude writers
    // (and FILE_FLAG_NO_BUFFERING below requires write sharing).
    file_ = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) { fail(path_, "open", ::GetLastError()); }

    LARGE_INTEGER file_size{};
    if (!::GetFileSizeEx(file_, &file_size)) {
        const auto error = ::GetLastError();
        ::CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
        fail(path_, "fstat", error);
    }
    bytes_ = static_cast<std::uint64_t>(file_size.QuadPart);
}

InputFile::~InputFile() {
    if (direct_file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(direct_file_); }
    if (file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(file_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read = read_at(path_, file_, offset, destination.subspan(0, count));
        if (read == 0) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += read;
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment) {
        throw ArtifactError(path_.string() + ": unaligned direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_file_ == INVALID_HANDLE_VALUE) {
        direct_file_ = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                     nullptr, OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                         FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                                     nullptr);
        if (direct_file_ == INVALID_HANDLE_VALUE) {
            fail(path_, "open direct", ::GetLastError());
        }
    }

    std::size_t total = 0;
    while (total < destination.size()) {
        constexpr std::size_t max_read = 1ULL << 30;
        const auto amount = static_cast<DWORD>(std::min<std::size_t>(max_read,
                                                                      destination.size() - total));
        const std::uint64_t absolute = offset + total;
        OVERLAPPED operation{};
        operation.Offset     = static_cast<DWORD>(absolute & 0xffffffffULL);
        operation.OffsetHigh = static_cast<DWORD>(absolute >> 32U);

        DWORD bytes = 0;
        const BOOL started =
            ::ReadFile(direct_file_, destination.data() + total, amount, &bytes, &operation);
        if (!started) {
            const auto error = ::GetLastError();
            if (error == ERROR_HANDLE_EOF) { break; }
            if (error != ERROR_IO_PENDING ||
                !::GetOverlappedResult(direct_file_, &operation, &bytes, TRUE)) {
                fail(path_, "direct read",
                     error == ERROR_IO_PENDING ? ::GetLastError() : error);
            }
        }
        total += bytes;
        if (bytes != amount) { break; }
    }
    return total;
}
#else
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

#endif  // _WIN32
} // namespace ninfer::artifact
