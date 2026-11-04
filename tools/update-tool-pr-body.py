#!/usr/bin/env python3

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

"""Write the description of the pull request that moves ruff or ty to a new release.

``update-tools.yml`` runs this once it has bumped the tool, formatted the tree and run the
hooks. The description says what moved and where to read about it, which files changed,
and, when hooks fail, which ones, with their output and links to the log, so that the
errors are one click away from the pull request.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# GitHub rejects descriptions over 65536 characters; the rest of the body is far shorter.
MAX_OUTPUT_CHARS = 45_000
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# prek prints `<hook name>.....<status>` per hook, `(no files to check)` before a skip,
# and a failing hook's id, exit code and output on the lines after its status line.
STATUS = re.compile(
    r"^(?P<name>\S.*?)\.{3,}(?:\([^)]*\))?(?P<status>Passed|Failed|Skipped)$"
)
HOOK_ID = re.compile(r"^- hook id: (?P<id>\S+)$")
# `just` reports the failing recipe after the hooks; it is not part of any hook's output.
JUST_ERROR = re.compile(r"^error: recipe `[^`]+` failed")
BUMPED_FILES = ("pyproject.toml", "prek.toml", "uv.lock")


def failing_hooks(log: str) -> list[tuple[str, str, str]]:
    """Return (name, id, output) for every failing hook in a ``prek run`` log, in order."""
    hooks: list[tuple[str, str, list[str]]] = []
    current: list[str] | None = None
    for raw in ANSI.sub("", log).splitlines():
        line = raw.rstrip()
        status = STATUS.match(line)
        if status:
            current = None
            if status["status"] == "Failed":
                current = [line]
                hooks.append((status["name"].strip(), "", current))
            continue
        if current is None or JUST_ERROR.match(line):
            continue
        hook_id = HOOK_ID.match(line)
        if hook_id and not hooks[-1][1]:
            hooks[-1] = (hooks[-1][0], hook_id["id"], current)
        current.append(line)
    return [
        (name, hook_id or name, "\n".join(out).strip()) for name, hook_id, out in hooks
    ]


def truncated(text: str, limit: int) -> tuple[str, bool]:
    """Cut ``text`` at the last line break before ``limit`` characters."""
    if len(text) <= limit:
        return text, False
    return text[: text.rfind("\n", 0, limit)], True


def fenced(text: str, language: str = "") -> str:
    """Fence ``text`` with more backticks than any run inside it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def release_links(tool: str, current: str, latest: str, releases: list[str]) -> str:
    """Link the release notes of every release since ``current`` and the full diff."""
    repo = f"https://github.com/astral-sh/{tool}"
    notes = ", ".join(f"[{v}]({repo}/releases/tag/{v})" for v in releases or [latest])
    return (
        f"- Release notes: {notes}\n"
        f"- [Changelog]({repo}/blob/main/CHANGELOG.md)"
        f" · [diff {current}...{latest}]({repo}/compare/{current}...{latest})"
    )


def changes(tool: str, current: str, latest: str, changed: list[str]) -> str:
    """Describe the files that the update changed."""
    described = {
        "pyproject.toml": f"`pyproject.toml`: `{tool}=={current}` → `{tool}=={latest}` in the"
        " `dev` dependency group.",
        "prek.toml": f"`prek.toml`: the `astral-sh/{tool}-pre-commit` hook from `v{current}`"
        f" to `v{latest}`, so that the hooks and `uv run {tool}` run the same release.",
        "uv.lock": f"`uv.lock`: refreshed with `uv lock --upgrade-package {tool}`.",
    }
    lines = [f"- {text}" for file, text in described.items() if file in changed]
    others = [file for file in changed if file not in BUMPED_FILES]
    source = "`just fmt` and the hooks" if tool == "ruff" else "the hooks"
    if others:
        listing = "\n".join(f"  - `{file}`" for file in others)
        lines.append(
            f"- {len(others)} more file{'s' if len(others) != 1 else ''} changed by"
            f" {source}:\n\n  <details><summary>Files</summary>\n\n{listing}\n\n  </details>",
        )
    elif tool == "ruff":
        lines.append(
            "- `just fmt` changed nothing: the new release formats the code as before."
        )
    return "\n".join(lines)


def checks(
    tool: str,
    latest: str,
    branch: str,
    hooks: list[tuple[str, str, str]],
    links: str,
    *,
    lint_failed: bool,
) -> str:
    """Report the outcome of the hooks, with the output of the failing ones."""
    if not lint_failed:
        return (
            f"✅ Every hook of `just lint` passes with {tool} {latest}.\n\n{links}\n\n"
            "The checks of this pull request run the tests and build the documentation."
        )

    if hooks:
        names = ", ".join(f"`{hook_id}`" for _, hook_id, _ in hooks)
        count = (
            f"{len(hooks)} hooks ({names}) fail"
            if len(hooks) > 1
            else f"The {names} hook fails"
        )
    else:
        count = "`just lint` fails"
    parts = [
        (
            "> [!WARNING]\n"
            f"> {count} with {tool} {latest}, so this pull request is a draft. Push the"
            f" fixes to `{branch}` and mark it ready for review."
        ),
        links,
    ]
    if hooks:
        output, cut = truncated(
            "\n\n".join(out for _, _, out in hooks),
            MAX_OUTPUT_CHARS,
        )
        note = (
            "\n\nThe output is truncated; the log above has all of it." if cut else ""
        )
        parts.append(
            "<details open><summary>Output of the failing hooks</summary>\n\n"
            f"{fenced(output, 'text')}{note}\n\n</details>",
        )
    else:
        parts.append(
            "No hook reports a failure; the log above shows where `just lint` stopped."
        )
    parts.append(
        "To reproduce it locally:\n\n"
        + fenced(
            f"gh pr checkout {branch}\njust build --all-groups --all-extras\njust lint",
            "bash",
        ),
    )
    return "\n\n".join(parts)


def main() -> int:
    """Write the pull request description to ``--output``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, choices=["ruff", "ty"])
    parser.add_argument("--current", required=True, help="the pinned release")
    parser.add_argument("--latest", required=True, help="the release to move to")
    parser.add_argument(
        "--releases",
        default="",
        help="space-separated releases after --current, up to --latest",
    )
    parser.add_argument("--branch", required=True, help="the pull request branch")
    parser.add_argument(
        "--lint-log", type=Path, required=True, help="output of `just lint`"
    )
    parser.add_argument(
        "--lint-outcome",
        required=True,
        choices=["success", "failure"],
    )
    parser.add_argument(
        "--changed-files",
        type=Path,
        required=True,
        help="file with one changed path per line",
    )
    parser.add_argument("--run-url", required=True, help="the workflow run")
    parser.add_argument("--job-url", required=True, help="the job's log")
    parser.add_argument("--hooks-url", required=True, help="the log of the hooks step")
    parser.add_argument("--workflow-url", required=True, help="the workflow's runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    log = args.lint_log.read_text() if args.lint_log.exists() else ""
    changed = [line for line in args.changed_files.read_text().splitlines() if line]
    links = (
        f"Logs: [workflow run]({args.run_url}) · [job]({args.job_url})"
        f" · [`just lint` step]({args.hooks_url})"
    )
    sections = [
        "## Summary",
        (
            f"Moves [{args.tool}](https://github.com/astral-sh/{args.tool}) from"
            f" {args.current} to {args.latest}: the pin of the `dev` group and the prek hook"
            " move together, so that local runs, the hooks and CI check the code with the"
            " same release."
        ),
        release_links(args.tool, args.current, args.latest, args.releases.split()),
        "## Changes",
        changes(args.tool, args.current, args.latest, changed),
        "## Checks",
        checks(
            args.tool,
            args.latest,
            args.branch,
            failing_hooks(log),
            links,
            lint_failed=args.lint_outcome == "failure",
        ),
        "---",
        (
            f"<sub>Opened by [Update ruff and ty]({args.workflow_url}) in"
            f" [this run]({args.run_url}).</sub>"
        ),
    ]
    args.output.write_text("\n\n".join(sections) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
