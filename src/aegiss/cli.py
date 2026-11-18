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

import sys
import tomllib
from pathlib import Path  # noqa: TC003 (cyclopts resolves the annotations at runtime)
from typing import TYPE_CHECKING

from cyclopts import App

from aegiss._version import __version__
from aegiss.utils import skill_bundle
from aegiss.utils.orb_display import (
    ORB_BASE_ENV_VAR,
    OrbitalBase,
    config_path,
    orbital_base,
    reset_cache,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

app = App(name="aegiss", help="AEGISS command-line tools.")
config_app = App(name="config", help="Inspect or change persisted settings.")
app.command(config_app)
skill_app = App(name="skill", help="Install the AEGISS agent skill.")
app.command(skill_app)


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


@config_app.command(
    name="set-base",
    help=(
        "Persist the base used when orbital indices are printed and "
        f"interpreted. Overridable per run via ${ORB_BASE_ENV_VAR}."
    ),
)
def _set_base(base: OrbitalBase, /) -> None:
    """Persist the orbital display base.

    Args:
        base: Index of the first orbital.
    """
    config = _load_config()
    config["orbital_base"] = base
    _write_config(config)
    reset_cache()
    print(f"Set orbital display base to {base} in {config_path()}.")


@config_app.command(name="get-base")
def _get_base() -> None:
    """Show the currently resolved base."""
    reset_cache()
    print(orbital_base())


@config_app.command(name="path")
def _print_path() -> None:
    """Show the config-file location."""
    print(config_path())


# The skill functions are called through the module so tests can swap the bundled skill.
@skill_app.command(name="path")
def _skill_path() -> None:
    """Show where the bundled skill lives."""
    print(skill_bundle.bundled_skill_dir())


@skill_app.command(name="install")
def _skill_install(
    directory: Path | None = None,
    /,
    *,
    agent: skill_bundle.Agent = "agents",
    user: bool = False,
    force: bool = False,
) -> int:
    """Copy the skill into an agent's skills folder, with docs links for this AEGISS version.

    Args:
        directory: Skills folder to install into; overrides ``--agent`` and ``--user``.
        agent: Agent family: ``agents`` (.agents/skills: pi, Codex, ...) or ``claude``
            (.claude/skills).
        user: Install in the home directory instead of the current project.
        force: Replace an installed copy.

    Returns:
        Process exit code: 1 when the skill is already installed and ``--force`` is not set.
    """
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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``aegiss`` console script.

    Invalid arguments print an error and raise ``SystemExit``.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    return app(argv, result_action="return_int_as_exit_code_else_zero")


if __name__ == "__main__":
    sys.exit(main())
