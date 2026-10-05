"""Shared pinned-artifact records and durable, resumable filesystem downloads."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import stat
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from errors import fail


@dataclass(frozen=True)
class Artifact:
    destination: Path
    role: str
    bytes: int
    sha256: str
    repository: str
    revision: str
    source_path: str

    @classmethod
    def from_entry(cls, destination: Path, entry: dict[str, Any], repository: str, revision: str) -> Artifact:
        return cls(destination, entry["role"], entry["bytes"], entry["sha256"],
                   repository, revision, str(entry.get("source_path", destination.name)))

    @property
    def source(self) -> str:
        return (f"https://huggingface.co/{self.repository}/resolve/{self.revision}/"
                + urllib.parse.quote(self.source_path, safe="/"))

    def entry(self) -> dict[str, Any]:
        return {"role": self.role, "bytes": self.bytes, "sha256": self.sha256,
                "source_path": self.source_path}

    def preview(self) -> dict[str, Any]:
        return {**self.entry(), "source": self.source, "repository": self.repository,
                "revision": self.revision, "destination": str(self.destination),
                "status": "verified-present" if file_matches(self.entry(), self.destination) else "download-required"}


def acquisition_plan(identifier: str, artifacts: tuple[Artifact, ...]) -> dict[str, Any]:
    files = [artifact.preview() for artifact in artifacts]
    return {"model": identifier, "files": files,
            "additional_download_bytes": sum(f["bytes"] for f in files if f["status"] == "download-required")}


def verify(artifacts: tuple[Artifact, ...], *, full: bool = False) -> list[str]:
    errors = []
    for artifact in artifacts:
        path = artifact.destination
        try:
            info = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_size != artifact.bytes:
                errors.append(f"missing or wrong size: {path}")
            elif full and sha256_file(path) != artifact.sha256:
                errors.append(f"checksum mismatch: {path}")
        except OSError:
            errors.append(f"missing or wrong size: {path}")
    return errors


def acquire(root: Path, artifacts: tuple[Artifact, ...]) -> None:
    for artifact in artifacts:
        download_huggingface_file(root, artifact.repository, artifact.revision,
                                 artifact.entry(), artifact.destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches(entry: dict[str, Any], path: Path) -> bool:
    try:
        info = path.stat(follow_symlinks=False)
        return (
            stat.S_ISREG(info.st_mode)
            and info.st_size == entry["bytes"]
            and sha256_file(path) == entry["sha256"]
        )
    except OSError:
        return False


def download_huggingface_file(
    root: Path, repository: str, revision: str, entry: dict[str, Any], destination: Path,
) -> None:
    root = root.resolve()
    try:
        destination.resolve().relative_to(root)
    except ValueError:
        fail(f"model download path escapes external model root: {destination}")
    if file_matches(entry, destination):
        print(f"Reusing verified artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    lock_path = destination.with_name(f".{destination.name}.lock")
    source_path = str(entry.get("source_path", destination.name))
    encoded_source_path = urllib.parse.quote(source_path, safe="/")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{encoded_source_path}"
    try:
        lock_fd = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
        )
    except OSError as exc:
        fail(f"cannot create safe model download lock {lock_path}: {exc}")
    with os.fdopen(lock_fd, "a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if file_matches(entry, destination):
            print(f"Reusing verified artifact: {destination}")
            return
        if partial.exists() and not stat.S_ISREG(partial.stat(follow_symlinks=False).st_mode):
            fail(f"refusing unsafe partial download path: {partial}")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > entry["bytes"]:
            partial.unlink()
            offset = 0
        if offset < entry["bytes"]:
            headers = {"User-Agent": "halo-ai/artifacts"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                response = urllib.request.urlopen(request, timeout=120)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                fail(f"download failed for {url}: {exc}")
            status_code = getattr(response, "status", response.getcode())
            append = offset > 0 and status_code == 206
            if offset and not append:
                offset = 0
            flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
            flags |= os.O_APPEND if append else os.O_TRUNC
            print(f"Downloading {repository}@{revision}/{destination.name} from byte {offset}")
            try:
                partial_fd = os.open(partial, flags, 0o640)
            except OSError as exc:
                response.close()
                fail(f"cannot create safe partial download {partial}: {exc}")
            with response, os.fdopen(partial_fd, "ab" if append else "wb") as stream:
                os.fchmod(stream.fileno(), 0o640)
                for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        actual_size = partial.stat().st_size if partial.exists() else 0
        actual_sha = sha256_file(partial) if actual_size == entry["bytes"] else ""
        if actual_size != entry["bytes"] or actual_sha != entry["sha256"]:
            with contextlib.suppress(FileNotFoundError):
                partial.unlink()
            fail(
                f"download verification failed for {destination.name}: "
                f"expected {entry['bytes']} bytes/{entry['sha256']}, got {actual_size} bytes/{actual_sha or 'not hashed'}"
            )
        os.replace(partial, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
