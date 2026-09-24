"""Keep large offline transfers from retaining whole artifacts in Linux's page cache."""

from __future__ import annotations

import os

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
_POSIX_PAGE_CACHE = hasattr(os, "sysconf") and hasattr(os, "posix_fadvise")
_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE") if _POSIX_PAGE_CACHE else 4096
if not _POSIX_PAGE_CACHE:
    # Windows: no page-cache advisory facilities
    os.POSIX_FADV_DONTNEED = 4
    os.posix_fadvise = lambda *args: None
    os.fdatasync = lambda fd: None


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if not _POSIX_PAGE_CACHE:
        return
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            if hasattr(os, "fdatasync"):
                os.fdatasync(fd)
            else:
                os.fsync(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0


def pwrite(fd: int, data: bytes | memoryview, offset: int) -> int:
    """POSIX os.pwrite; single-writer lseek+write fallback on Windows."""
    if hasattr(os, "pwrite"):
        return os.pwrite(fd, data, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.write(fd, data)


def pread(fd: int, count: int, offset: int) -> bytes:
    """POSIX os.pread; single-reader lseek+read fallback on Windows."""
    if hasattr(os, "pread"):
        return os.pread(fd, count, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, count)
