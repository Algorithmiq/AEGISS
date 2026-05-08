"""Tests for aegiss.utils.env — environment variable reader functions."""

from __future__ import annotations

import pytest

from aegiss.utils.env import omp_num_threads, pyscf_max_memory, pyscf_scratch_dir, scratch_dir


class TestOmpNumThreads:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        assert omp_num_threads() == 4

    def test_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        assert omp_num_threads() == 8

    def test_returns_int(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        assert isinstance(omp_num_threads(), int)


class TestScratchDir:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("SCRATCH_DIR", raising=False)
        assert scratch_dir() == "./tmp"

    def test_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("SCRATCH_DIR", "/custom/scratch")
        assert scratch_dir() == "/custom/scratch"


class TestPyscfMaxMemory:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("PYSCF_MAX_MEMORY", raising=False)
        assert pyscf_max_memory() == 4000

    def test_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "16000")
        assert pyscf_max_memory() == 16000

    def test_returns_int(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "2000")
        assert isinstance(pyscf_max_memory(), int)


class TestPyscfScratchDir:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("PYSCF_SCRATCH_DIR", raising=False)
        assert pyscf_scratch_dir() == "./tmp"

    def test_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_SCRATCH_DIR", "/pyscf/tmp")
        assert pyscf_scratch_dir() == "/pyscf/tmp"
