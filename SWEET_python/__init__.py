"""SWEET_python: the waste model shared by the Climate TRACE pipeline and WasteMAP.

WHAT ``__version__`` IS, AND WHY IT IS A SHA
--------------------------------------------
It is the commit this copy was installed from -- not a hand-maintained number, and not
the release name.

SWEET's releases are git tags, ``sweet/YYYY.MM.N``, cut from ``main`` by
``.github/workflows/release.yml`` (the README's Releases section has the scheme). A
release names exactly one commit and is never moved, so a release's commit says as much
as its name -- and the commit is what an installed copy can report on its own, with no
repository and no network, and the only name a copy installed from a branch has. A copy
installed from ``sweet/2026.10.0`` reports that tag's commit; to turn a sha from a log
back into a release name, ``git tag --points-at <sha> --list 'sweet/*'``. So there is
still no version number in this file. There is a build stamp.

WHAT IT IS FOR. The live DST endpoints (``/sdst``, ``/adst``, ``/cdst``) call this
package at request time, which makes them the one place in the system where SWEET's
behaviour is user-visible with no run and no manifest behind it. Until now nothing could
answer "which model is the live site running" -- the API knew the *ref* it asked for, and
the ref could be a lie, because the images fell back to ``main`` when it did not resolve.
A process that imports this can now log what it actually got.

HOW IT RESOLVES, in order, inventing nothing:

1. The build stamp, ``SWEET_python/_build_sha.py``: ``setup.py`` writes the commit into
   every copy built from a clean git checkout -- which is how ``pip install
   git+https://...@sweet/2026.10.0`` builds -- so the copy itself knows. First because
   it describes the installed code, where an environment variable is a claim made from
   outside it.
2. ``SWEET_GIT_SHA``, which the apps set from the commit they install, for a copy with
   no stamp (an editable install, or a path dependency).
3. A ``.git`` directory beside the package, for an editable checkout.
4. ``"unknown"`` -- an honest answer, and the one a reader must be able to distinguish
   from a real sha. ``VERSION_SOURCE`` says which of the four it was: ``build``,
   ``env``, ``git`` or ``unavailable``.

Same shape, deliberately, as the Climate TRACE pipeline's
``results_tables.resolve_git_shas``: env first, then a repository read, then a recorded
absence.

``VERSIONING.md`` in RMI_Climate_TRACE_Waste_Methane covers the identifiers around this
one: a run's ``run_id`` and the ``wastemap/`` and ``submission/`` tags.
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


def _sha_from_build() -> Optional[str]:
    try:
        from SWEET_python._build_sha import GIT_SHA
    except ImportError:
        return None
    return (GIT_SHA or "").strip() or None


def _resolve_version() -> "tuple[str, str]":
    sha = _sha_from_build()
    if sha:
        return sha, "build"
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
