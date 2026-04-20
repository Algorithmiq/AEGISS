# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-20

### Added
- Active space selection via AO projection (`active_space/ao_projection.py`)
- Entropy-based orbital screening (`active_space/entropy_screening.py`)
- DMRG-based orbital entanglement and entropy computation (`utils/dmrg_utils.py`)
- Natural orbital localization and frozen natural orbital (FNO) methods (`localization/`)
- Orbital index utilities (`utils/orbital_indices.py`)
- Intermediate dumping to cube files via pathlib-safe paths (`utils/dump_intermediates.py`)
- `main.py` entry point orchestrating the full AEGISS pipeline
- Test suite covering SCF, DMRG utils, entropy screening, orbital indices, and AO projection
- CI workflows for linting (ruff, mypy) and testing (pytest + coverage) on Python 3.12
- Pre-commit hooks for code quality enforcement
- README with installation guide, 6-step tutorial, and API reference

[Unreleased]: https://github.com/Algorithmiq/AEGISS/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Algorithmiq/AEGISS/releases/tag/v0.1.0
