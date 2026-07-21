"""Command-line entry point for AEGISS configuration.

Installed as the ``aegiss`` console script (see ``[project.scripts]`` in
``pyproject.toml``). Its main purpose is the one-time, post-install step of
choosing the base used when orbital indices are printed and interpreted::

    aegiss config set-base 1     # print/read orbital indices 1-based
    aegiss config set-base 0     # back to the 0-based default
    aegiss config get-base       # show the currently resolved base
    aegiss config path           # show the config-file location

The value is persisted to :func:`aegiss.utils.orb_display.config_path`. It can
always be overridden for a single run with the ``AEGISS_ORB_BASE`` environment
variable, which takes precedence over the file.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from typing import TYPE_CHECKING

from aegiss.utils.orb_display import (
    _ENV_VAR,
    _VALID_BASES,
    config_path,
    orbital_base,
    reset_cache,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _load_config() -> dict[str, object]:
    """Return the persisted config as a dict (empty if the file is absent)."""
    path = config_path()
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return dict(tomllib.load(handle))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"Warning: could not read config file {path}: {exc}", file=sys.stderr)
        return {}


def _dump_scalar(value: object) -> str:
    """Serialise a scalar to its TOML representation."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_config(data: dict[str, object]) -> None:
    """Persist *data* as a flat TOML file, creating parent directories.

    Only flat top-level scalar keys are supported, which is all the AEGISS
    config schema uses.
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AEGISS configuration. Managed by `aegiss config`.",
        *(f"{key} = {_dump_scalar(val)}" for key, val in sorted(data.items())),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _set_base(base: int) -> int:
    """Persist the orbital display *base*; return a process exit code."""
    config = _load_config()
    config["orbital_base"] = base
    _write_config(config)
    reset_cache()
    print(f"Set orbital display base to {base} in {config_path()}.")
    return 0


def _get_base() -> int:
    """Print the currently resolved orbital display base; return exit code."""
    reset_cache()
    print(orbital_base())
    return 0


def _print_path() -> int:
    """Print the config-file path; return exit code."""
    print(config_path())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``aegiss`` argument parser."""
    parser = argparse.ArgumentParser(prog="aegiss", description="AEGISS configuration.")
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("config", help="Inspect or change persisted settings.")
    config_sub = config.add_subparsers(dest="config_command", required=True)

    set_base = config_sub.add_parser(
        "set-base",
        help=(
            "Persist the base used when orbital indices are printed and "
            f"interpreted. Overridable per run via ${_ENV_VAR}."
        ),
    )
    set_base.add_argument("base", type=int, choices=_VALID_BASES)

    config_sub.add_parser("get-base", help="Show the currently resolved base.")
    config_sub.add_parser("path", help="Show the config-file location.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``aegiss`` console script.

    Args:
        argv (Sequence[str] or None): Argument vector; defaults to ``sys.argv``.

    Returns:
        int: Process exit code.
    """
    args = build_parser().parse_args(argv)
    if args.command == "config":
        if args.config_command == "set-base":
            return _set_base(args.base)
        if args.config_command == "get-base":
            return _get_base()
        if args.config_command == "path":
            return _print_path()
    return 1


if __name__ == "__main__":
    sys.exit(main())
