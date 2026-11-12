"""Storage abstraction: local filesystem (dev) and S3-compatible (prod).

Both drivers expose the same interface, and every key is namespaced per user
so that authorisation can be enforced before a signed URL is issued.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .config import settings


@dataclass
class StoredObject:
    key: str
    size: int
    sha256: str
    content_type: str


class Storage:
    """Interface implemented by every storage driver."""

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> StoredObject:
        raise NotImplementedError

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> StoredObject:
        raise NotImplementedError

    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def local_path(self, key: str) -> Path | None:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def delete_prefix(self, prefix: str) -> int:
        raise NotImplementedError

    def signed_url(self, key: str, ttl: int | None = None) -> str:
        raise NotImplementedError

    def size_of_prefix(self, prefix: str) -> int:
        raise NotImplementedError


class LocalStorage(Storage):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        key = key.lstrip("/")
        if ".." in Path(key).parts:
            raise ValueError("invalid storage key")
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError("path traversal blocked")
        return p

    def put_bytes(self, key, data, content_type="application/octet-stream") -> StoredObject:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return StoredObject(key, len(data), hashlib.sha256(data).hexdigest(), content_type)

    def put_file(self, key, path, content_type="application/octet-stream") -> StoredObject:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, p)
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return StoredObject(key, p.stat().st_size, digest.hexdigest(), content_type)

    def read_bytes(self, key) -> bytes:
        return self._path(key).read_bytes()

    def local_path(self, key) -> Path | None:
        return self._path(key)

    def exists(self, key) -> bool:
        return self._path(key).exists()

    def delete_prefix(self, prefix) -> int:
        p = self._path(prefix)
        removed = 0
        if p.is_dir():
            removed = sum(1 for _ in p.rglob("*") if _.is_file())
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
            removed = 1
        return removed

    def signed_url(self, key, ttl=None) -> str:
        return f"/api/v1/files/{key.lstrip('/')}"

    def size_of_prefix(self, prefix) -> int:
        p = self._path(prefix)
        if not p.exists():
            return 0
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class S3Storage(Storage):
    """S3 / R2 / MinIO driver using presigned URLs (boto3)."""

    def __init__(self) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = settings.storage_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.storage_endpoint or None,
            aws_access_key_id=settings.storage_access_key or None,
            aws_secret_access_key=settings.storage_secret_key or None,
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )

    def put_bytes(self, key, data, content_type="application/octet-stream") -> StoredObject:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return StoredObject(key, len(data), hashlib.sha256(data).hexdigest(), content_type)

    def put_file(self, key, path, content_type="application/octet-stream") -> StoredObject:
        self.client.upload_file(str(path), self.bucket, key,
                                ExtraArgs={"ContentType": content_type})
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return StoredObject(key, Path(path).stat().st_size, digest.hexdigest(), content_type)

    def read_bytes(self, key) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def local_path(self, key) -> Path | None:
        return None

    def exists(self, key) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def delete_prefix(self, prefix) -> int:
        paginator = self.client.get_paginator("list_objects_v2")
        removed = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})
                removed += len(keys)
        return removed

    def signed_url(self, key, ttl=None) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=ttl or settings.signed_url_ttl)

    def size_of_prefix(self, prefix) -> int:
        paginator = self.client.get_paginator("list_objects_v2")
        total = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            total += sum(o["Size"] for o in page.get("Contents", []))
        return total


def _build_storage() -> Storage:
    if settings.storage_driver == "s3":
        return S3Storage()
    return LocalStorage(settings.storage_root)


storage: Storage = _build_storage()


def content_type_for(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def user_prefix(user_id: str) -> str:
    return f"users/{user_id}"


def upload_key(user_id: str, upload_id: str, filename: str, original: bool = True) -> str:
    suffix = Path(filename).suffix.lower() or ".bin"
    kind = "original" if original else "thumb"
    return f"{user_prefix(user_id)}/uploads/{upload_id}/{kind}{suffix}"


def generation_key(user_id: str, generation_id: str, relpath: str) -> str:
    return f"{user_prefix(user_id)}/generations/{generation_id}/{relpath.lstrip('/')}"


def asset_prefix(user_id: str, generation_id: str, asset_id: str) -> str:
    return f"{user_prefix(user_id)}/generations/{generation_id}/assets/{asset_id}"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
