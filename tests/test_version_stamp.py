"""``SWEET_python.__version__`` is the commit this copy came from, or it says it cannot tell.

There is no hand-maintained SWEET version; see the package docstring for why. What there
is has one job: let a process that imported this package say which model it is running.
The live DST endpoints are the reason -- they call SWEET at request time, so they are the
one place where its behaviour ships with no run manifest behind it.

The rule these tests exist to hold: it never invents a value. A wrong sha in a log is
worse than none, because a reader believes it.
"""

from __future__ import annotations

import importlib
import subprocess

import pytest

import SWEET_python


def _reimport(monkeypatch, **env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return importlib.reload(SWEET_python)


def test_an_env_sha_wins(monkeypatch):
    """The image build knows the ref it actually installed and passes it in. That beats
    reading a repository, because the image holds source, not a repository."""
    mod = _reimport(monkeypatch, SWEET_GIT_SHA="a" * 40)

    assert mod.__version__ == "a" * 40
    assert mod.VERSION_SOURCE == "env"


def test_a_blank_env_sha_does_not_count(monkeypatch):
    """An unset variable arrives as an empty string often enough that it has to be
    handled: a deploy that forgot to pass one must fall through, not stamp ''."""
    mod = _reimport(monkeypatch, SWEET_GIT_SHA="   ")

    assert mod.__version__ != ""
    assert mod.VERSION_SOURCE in {"git", "unavailable"}


def test_an_editable_checkout_reads_its_own_git(monkeypatch):
    mod = _reimport(monkeypatch, SWEET_GIT_SHA=None)

    if mod.VERSION_SOURCE == "unavailable":
        pytest.skip("no .git beside the package; nothing to compare against")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert mod.__version__ == head
    assert mod.VERSION_SOURCE == "git"


def test_an_unreadable_install_says_unknown_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(SWEET_python, "_sha_from_git", lambda _: None)
    monkeypatch.delenv("SWEET_GIT_SHA", raising=False)

    version, source = SWEET_python._resolve_version()

    assert version == "unknown"
    assert source == "unavailable"


def test_version_info_carries_the_source(monkeypatch):
    """A bare "unknown" in a log reads as a missing feature. "unknown (unavailable)"
    reads as an install that cannot say what it is, which is the true statement."""
    info = SWEET_python.version_info()

    assert set(info) == {"sweet_version", "sweet_version_source"}
    assert info["sweet_version"] == SWEET_python.__version__
    assert info["sweet_version_source"] == SWEET_python.VERSION_SOURCE


def test_importing_the_package_costs_no_subprocess_without_a_git_dir(monkeypatch):
    """The `.git` existence check comes first on purpose. In the container there is no
    repository, and SWEET is imported by every multiprocessing worker."""
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))

    assert SWEET_python._sha_from_git(pytest.importorskip("pathlib").Path("/nonexistent")) is None
    assert calls == []
