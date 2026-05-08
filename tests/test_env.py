"""Tests for aegiss.utils.env — environment variable reader functions."""

from __future__ import annotations

import pytest

from aegiss.utils.env import omp_num_threads, pyscf_max_memory, scratch_dir


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

    def test_invalid_value(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "not_an_int")
        with pytest.raises(
            ValueError, match="Invalid value for OMP_NUM_THREADS: 'not_an_int'"
        ):
            omp_num_threads()

    def test_empty_value(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "")
        assert omp_num_threads() == 4

    def test_whitespace_value(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "   ")
        assert omp_num_threads() == 4


class TestScratchDir:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("SCRATCH_DIR", raising=False)
        assert scratch_dir() == "./tmp"

    def test_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("SCRATCH_DIR", "/custom/scratch")
        assert scratch_dir() == "/custom/scratch"

    def test_empty_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("SCRATCH_DIR", "")
        assert scratch_dir() == "./tmp"

    def test_whitespace_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("SCRATCH_DIR", "   ")
        assert scratch_dir() == "./tmp"


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

    def test_invalid_value(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "not_an_int")
        with pytest.raises(
            ValueError, match="Invalid value for PYSCF_MAX_MEMORY: 'not_an_int'"
        ):
            pyscf_max_memory()

    def test_empty_value(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "")
        assert pyscf_max_memory() == 4000

    def test_whitespace_value(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "   ")
        assert pyscf_max_memory() == 4000
