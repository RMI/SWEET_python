"""SWEET_python: the waste model shared by the Climate TRACE pipeline and WasteMAP.

WHAT ``__version__`` IS, AND WHY IT IS A SHA
--------------------------------------------
It is the commit this copy was installed from -- not a hand-maintained number.

SWEET never ships on its own. It reaches the world only inside a Climate TRACE run or a
WasteMAP deploy, and both of those already have identities of their own (a ``run_id`` and
a ``wastemap/YYYY.MM.DD`` tag). A ``MAJOR.MINOR`` maintained by hand here would be a
second, weaker name for a commit that is already recorded by sha in every run manifest,
and it would need bumping on every model change by the same person who has to remember
the tag. So there is no SWEET version number. There is a build stamp.

WHAT IT IS FOR. The live DST endpoints (``/sdst``, ``/adst``, ``/cdst``) call this
package at request time, which makes them the one place in the system where SWEET's
behaviour is user-visible with no run and no manifest behind it. Until now nothing could
answer "which model is the live site running" -- the API knew the *ref* it asked for, and
the ref could be a lie, because the images fell back to ``main`` when it did not resolve.
A process that imports this can now log what it actually got.

HOW IT RESOLVES, in order, inventing nothing:

1. ``SWEET_GIT_SHA``, which the image build sets from the ref it actually installed.
2. A ``.git`` directory beside the package, for an editable checkout.
3. ``"unknown"`` -- an honest answer, and the one a reader must be able to distinguish
   from a real sha. ``VERSION_SOURCE`` says which of the three it was.

Same shape, deliberately, as the Climate TRACE pipeline's
``results_tables.resolve_git_shas``: env first, then a repository read, then a recorded
absence.

See ``VERSIONING.md`` in RMI_Climate_TRACE_Waste_Methane for the whole scheme.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

__all__ = ["__version__", "VERSION_SOURCE", "version_info"]


def _sha_from_git(repo_dir: Path) -> Optional[str]:
    if not (repo_dir / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    return (out.stdout or "").strip() or None


def _resolve_version() -> "tuple[str, str]":
    sha = (os.environ.get("SWEET_GIT_SHA") or "").strip()
    if sha:
        return sha, "env"
    sha = _sha_from_git(Path(__file__).resolve().parents[1])
    if sha:
        return sha, "git"
    return "unknown", "unavailable"


__version__, VERSION_SOURCE = _resolve_version()


def version_info() -> dict:
    """``{"sweet_version": ..., "sweet_version_source": ...}``, for a startup log line.

    A dict rather than a string so a caller logging it cannot drop the source: a bare
    ``"unknown"`` in a log reads as a missing feature, while ``unknown (unavailable)``
    reads as what it is -- an install that cannot say what it is.
    """
    return {"sweet_version": __version__, "sweet_version_source": VERSION_SOURCE}
