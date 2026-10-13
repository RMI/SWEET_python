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


def test_an_env_sha_wins_over_a_repository(monkeypatch):
    """With no build stamp, the app's SWEET_GIT_SHA beats reading a repository, because
    an image holds source, not a repository."""
    monkeypatch.setattr(SWEET_python, "_sha_from_build", lambda: None)
    monkeypatch.setenv("SWEET_GIT_SHA", "a" * 40)

    assert SWEET_python._resolve_version() == ("a" * 40, "env")


def test_a_blank_env_sha_does_not_count(monkeypatch):
    """An unset variable arrives as an empty string often enough that it has to be
    handled: a deploy that forgot to pass one must fall through, not stamp ''."""
    mod = _reimport(monkeypatch, SWEET_GIT_SHA="   ")

    assert mod.__version__ != ""
    assert mod.VERSION_SOURCE in {"build", "git", "unavailable"}


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


def test_a_build_stamp_beats_the_environment(monkeypatch):
    """A copy built from a git checkout carries its commit (setup.py writes
    _build_sha.py). That describes the installed code; an env var is a claim from outside."""
    monkeypatch.setattr(SWEET_python, "_sha_from_build", lambda: "b" * 40)
    monkeypatch.setenv("SWEET_GIT_SHA", "a" * 40)

    assert SWEET_python._resolve_version() == ("b" * 40, "build")


def test_without_a_stamp_the_environment_still_counts(monkeypatch):
    monkeypatch.setattr(SWEET_python, "_sha_from_build", lambda: None)
    monkeypatch.setenv("SWEET_GIT_SHA", "a" * 40)

    assert SWEET_python._resolve_version() == ("a" * 40, "env")


def test_the_source_tree_carries_no_stamp():
    """setup.py writes the stamp into the build directory only, so an editable install
    of a checkout can never report a stale commit from one."""
    from pathlib import Path

    assert not (Path(SWEET_python.__file__).parent / "_build_sha.py").exists()


def test_setup_stamps_only_a_clean_checkout(tmp_path):
    """The commit a build reports is the code it built: a dirty tree gets no stamp."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "sweet_setup", Path(SWEET_python.__file__).parents[1] / "setup.py"
    )
    setup_src = spec.origin
    namespace: dict = {}
    # Only the helper, not setup() itself.
    source = Path(setup_src).read_text()
    helper = source[source.index("def _clean_head") : source.index("class build_py_with_commit")]
    exec("import pathlib, subprocess\n" + helper, namespace)
    clean_head = namespace["_clean_head"]

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    (tmp_path / "f.txt").write_text("1")
    git("add", "f.txt")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "y")
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    assert clean_head(tmp_path) == head
    (tmp_path / "f.txt").write_text("2")
    assert clean_head(tmp_path) is None
    assert clean_head(tmp_path / "not-a-repo") is None


def test_an_unreadable_install_says_unknown_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(SWEET_python, "_sha_from_build", lambda: None)
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
