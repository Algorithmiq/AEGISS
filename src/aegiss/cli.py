# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line entry point for AEGISS.

Installed as the ``aegiss`` console script (see ``[project.scripts]`` in
``pyproject.toml``). It chooses the base used when orbital indices are printed
and interpreted, and installs the AEGISS agent skill::

    aegiss config set-base 1     # print/read orbital indices 1-based
    aegiss config set-base 0     # back to the 0-based default
    aegiss config get-base       # show the currently resolved base
    aegiss config path           # show the config-file location
    aegiss skill install         # install the agent skill in ./.agents/skills
    aegiss skill path            # show where the bundled skill lives

The value is persisted to :func:`aegiss.utils.orb_display.config_path`. It can
always be overridden for a single run with the ``AEGISS_ORB_BASE`` environment
variable, which takes precedence over the file.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from aegiss._version import __version__
from aegiss.utils import skill_bundle
from aegiss.utils.orb_display import (
    ORB_BASE_ENV_VAR,
    VALID_BASES,
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


def _skill_path() -> int:
    """Print where the bundled skill lives; return exit code."""
    print(skill_bundle.bundled_skill_dir())
    return 0


def _skill_install(
    directory: Path | None, agent: str, *, user: bool, force: bool
) -> int:
    """Install the bundled skill; return exit code."""
    skills_dir = directory or skill_bundle.skills_root(agent, user=user)
    try:
        target = skill_bundle.install_skill(skills_dir, force=force)
    except FileExistsError as exc:
        print(
            f"{exc.filename} already exists; rerun with --force to replace it.",
            file=sys.stderr,
        )
        return 1
    print(f"Installed the AEGISS {__version__} skill in {target}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``aegiss`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aegiss", description="AEGISS command-line tools."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("config", help="Inspect or change persisted settings.")
    config_sub = config.add_subparsers(dest="config_command", required=True)

    set_base = config_sub.add_parser(
        "set-base",
        help=(
            "Persist the base used when orbital indices are printed and "
            f"interpreted. Overridable per run via ${ORB_BASE_ENV_VAR}."
        ),
    )
    set_base.add_argument("base", type=int, choices=VALID_BASES)

    config_sub.add_parser("get-base", help="Show the currently resolved base.")
    config_sub.add_parser("path", help="Show the config-file location.")

    skill = sub.add_parser("skill", help="Install the AEGISS agent skill.")
    skill_sub = skill.add_subparsers(dest="skill_command", required=True)
    skill_sub.add_parser("path", help="Show where the bundled skill lives.")
    install = skill_sub.add_parser(
        "install",
        help=(
            "Copy the skill into an agent's skills folder, with docs links for "
            "this AEGISS version."
        ),
    )
    install.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Skills folder to install into; overrides --agent and --user.",
    )
    install.add_argument(
        "--agent",
        choices=sorted(skill_bundle.AGENT_DIRS),
        default="agents",
        help="'agents' (.agents/skills: pi, Codex, ...) or 'claude' (.claude/skills).",
    )
    install.add_argument(
        "--user", action="store_true", help="Install in the home directory."
    )
    install.add_argument(
        "--force", action="store_true", help="Replace an installed copy."
    )
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
    if args.command == "skill":
        if args.skill_command == "path":
            return _skill_path()
        if args.skill_command == "install":
            return _skill_install(
                args.directory, args.agent, user=args.user, force=args.force
            )
    return 1


if __name__ == "__main__":
    sys.exit(main())
