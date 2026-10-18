"""Registry persistence behind an interface.

Two implementations ship with the platform:

- ``InMemoryRegistryStore`` — tests and local development.
- ``JsonFileRegistryStore`` — durable, zero-infrastructure persistence that
  survives process restarts (a single JSON document, rewritten atomically).

A production deployment implements the same ``RegistryStore`` protocol
against its database or managed ML platform.
"""
from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict

from .models import ModelVersion, Stage


def _to_record(mv: ModelVersion) -> dict:
    rec = asdict(mv)
    rec["stage"] = mv.stage.value
    return rec


def _from_record(rec: dict) -> ModelVersion:
    rec = dict(rec)
    rec["stage"] = Stage(rec["stage"])
    return ModelVersion(**rec)


class RegistryStore(ABC):
    """Persistence contract for model versions."""

    @abstractmethod
    def save(self, mv: ModelVersion) -> None:
        ...

    @abstractmethod
    def get(self, name: str, version: str) -> ModelVersion | None:
        ...

    @abstractmethod
    def list(self, name: str | None = None) -> list[ModelVersion]:
        ...


class InMemoryRegistryStore(RegistryStore):
    def __init__(self) -> None:
        self._data: dict[str, ModelVersion] = {}

    def save(self, mv: ModelVersion) -> None:
        self._data[mv.key] = mv

    def get(self, name: str, version: str) -> ModelVersion | None:
        return self._data.get(f"{name}@{version}")

    def list(self, name: str | None = None) -> list[ModelVersion]:
        out = [mv for mv in self._data.values() if name is None or mv.name == name]
        return sorted(out, key=lambda m: (m.name, m.created_at))


class JsonFileRegistryStore(RegistryStore):
    def __init__(self, path: str) -> None:
        self._path = path
        self._data: dict[str, ModelVersion] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for rec in json.load(fh):
                    mv = _from_record(rec)
                    self._data[mv.key] = mv

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        records = [_to_record(mv) for mv in self._data.values()]
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self._path)))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)
            os.replace(tmp, self._path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def save(self, mv: ModelVersion) -> None:
        self._data[mv.key] = mv
        self._flush()

    def get(self, name: str, version: str) -> ModelVersion | None:
        return self._data.get(f"{name}@{version}")

    def list(self, name: str | None = None) -> list[ModelVersion]:
        out = [mv for mv in self._data.values() if name is None or mv.name == name]
        return sorted(out, key=lambda m: (m.name, m.created_at))
