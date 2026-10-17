#!/usr/bin/env python3
"""Name SWEET's next release, check a release name, or find the latest one.

A SWEET release is a git tag ``sweet/YYYY.MM.N``: the UTC year and month it was cut,
then a counter that starts at 0 each month. ``.github/workflows/release.yml`` is the
only caller; it lives here, rather than inline in the workflow, so the arithmetic has
tests (``tests/test_next_release.py``) and no network is needed to run them.

Tag names come in on stdin, one per line -- whatever ``git tag --list`` printed -- so
this never runs git itself and a test can hand it any list it likes:

    git tag --list 'sweet/*' | scripts/next_release.py           # next name this month
    git tag --list 'sweet/*' | scripts/next_release.py 2026.10   # next name in 2026-10
    scripts/next_release.py --check sweet/2026.10.3              # is this a release name?
    git tag --merged HEAD | scripts/next_release.py --latest     # newest release in a list

Anything on stdin that is not a release name is ignored, not an error: the repository
also carries other projects' tags (``wastemap/2026.10.07``, ``submission/08_03_26``,
``v2026.08.17``), and a hand-pushed ``sweet/test`` must not stop a real release.

Standard library only, so the workflow can run it with the runner's own ``python3``
before any dependencies are installed.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple

# The pattern the workflow documents, with [0-9] rather than \d (\d also matches
# non-ASCII digits) and no leading zeros on N, so each release has exactly one
# spelling: sweet/2026.10.1 and sweet/2026.10.01 cannot both exist.
RELEASE_NAME = re.compile(r"sweet/([0-9]{4})\.([0-9]{2})\.(0|[1-9][0-9]*)")
MONTH = re.compile(r"([0-9]{4})\.([0-9]{2})")


def parse(name: str) -> Optional[Tuple[int, int, int]]:
    """``(year, month, n)`` for a release name, or ``None`` for anything else.

    The month must be 01-12 as well as two digits: ``sweet/2026.13.0`` matches the
    documented pattern but can only be a typo, and a release named for a month that
    does not exist would sort after every real one.
    """
    m = RELEASE_NAME.fullmatch(name.strip())
    if not m:
        return None
    year, month, n = (int(g) for g in m.groups())
    if not 1 <= month <= 12:
        return None
    return year, month, n


def next_release(tags: Iterable[str], month: str) -> str:
    """The next free release name in ``month`` (``YYYY.MM``), given existing tags.

    One more than the highest N already used that month, not the count of names:
    if a gap ever exists (someone named ``.3`` by hand after ``.0`` and ``.1``), a
    count would hand out ``.3`` again, and a release must always sort after the ones
    before it.
    """
    m = MONTH.fullmatch(month.strip())
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise ValueError(f"month must be YYYY.MM, got {month!r}")
    year, mon = int(m.group(1)), int(m.group(2))
    used = [p[2] for p in map(parse, tags) if p and p[:2] == (year, mon)]
    n = max(used) + 1 if used else 0
    return f"sweet/{year:04d}.{mon:02d}.{n}"


def latest_release(tags: Iterable[str]) -> Optional[str]:
    """The highest release name in ``tags``, by (year, month, N), or ``None``.

    Compared as numbers, not strings: as text, ``sweet/2026.10.10`` sorts before
    ``sweet/2026.10.9``.
    """
    releases = [(p, t.strip()) for t in tags if (p := parse(t))]
    return max(releases)[1] if releases else None


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y.%m")


def main(argv: list[str]) -> int:
    if argv[:1] == ["--check"]:
        if len(argv) != 2:
            print("usage: next_release.py --check NAME", file=sys.stderr)
            return 2
        name = argv[1].strip()
        if parse(name) is None:
            print(
                f"{argv[1]!r} is not a release name: expected sweet/YYYY.MM.N, "
                "with a real month and no leading zero on N (e.g. sweet/2026.10.0)",
                file=sys.stderr,
            )
            return 1
        # Printed back trimmed, so a name pasted with a stray space is used as the
        # tag it was meant to be rather than refused for the space.
        print(name)
        return 0

    tags = sys.stdin.read().splitlines()

    if argv[:1] == ["--latest"]:
        latest = latest_release(tags)
        if latest:
            print(latest)
        return 0

    if len(argv) > 1:
        print("usage: next_release.py [YYYY.MM]", file=sys.stderr)
        return 2
    try:
        print(next_release(tags, argv[0] if argv else _current_month()))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
