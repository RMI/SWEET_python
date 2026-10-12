"""``scripts/next_release.py`` names SWEET's releases; these hold its arithmetic.

The release workflow pipes ``git tag --list`` into the script and tags whatever comes
back, and a release tag is never moved or reused -- so a wrong name here is a wrong
name forever. Every test hands the script a tag list directly: no git, no network.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "next_release.py"

# Loaded by path, not imported as ``scripts.next_release``: ``scripts/`` is not a
# package, and in an environment that also has WasteMAP installed editable, the name
# ``scripts`` already belongs to WasteMAP.
_spec = importlib.util.spec_from_file_location("next_release", SCRIPT)
next_release_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(next_release_mod)

next_release = next_release_mod.next_release
latest_release = next_release_mod.latest_release

# Tags this repository really carries that belong to other projects.
OTHER_PROJECTS = [
    "v2026.08.17",
    "wastemap/2026.08.17",
    "wastemap/2026.10.07",
    "submission/08_03_26",
]


def test_first_release_of_a_month_is_zero():
    assert next_release([], "2026.10") == "sweet/2026.10.0"
    assert next_release(OTHER_PROJECTS, "2026.10") == "sweet/2026.10.0"


def test_after_zero_and_one_comes_two():
    assert next_release(["sweet/2026.10.0", "sweet/2026.10.1"], "2026.10") == "sweet/2026.10.2"


def test_order_of_the_tag_list_does_not_matter():
    assert next_release(["sweet/2026.10.1", "sweet/2026.10.0"], "2026.10") == "sweet/2026.10.2"


def test_counting_is_numeric_past_nine():
    tags = [f"sweet/2026.10.{n}" for n in range(11)]
    assert next_release(tags, "2026.10") == "sweet/2026.10.11"


def test_releases_from_other_months_are_ignored():
    tags = [
        "sweet/2026.09.0",
        "sweet/2026.09.1",
        "sweet/2026.09.2",
        "sweet/2025.10.4",  # same month, another year
        "sweet/2026.11.0",
    ]
    assert next_release(tags, "2026.10") == "sweet/2026.10.0"
    assert next_release(tags + ["sweet/2026.10.0"], "2026.10") == "sweet/2026.10.1"


@pytest.mark.parametrize(
    "malformed",
    [
        "sweet/2026.10.01",  # leading zero: a second spelling of .1
        "sweet/2026.10",
        "sweet/2026.10.",
        "sweet/2026.10.x",
        "sweet/2026.10.7-rc1",
        "sweet/2026.1.7",
        "sweet/26.10.7",
        "sweet/test",
        "SWEET/2026.10.7",
        "release/2026.10.7",
        "sweet/2026.10.7/extra",
        "sweet/2026.10.٧",  # a non-ASCII digit, which \d would have accepted
    ],
)
def test_malformed_tags_are_ignored(malformed):
    assert next_release(["sweet/2026.10.0", malformed], "2026.10") == "sweet/2026.10.1"


def test_a_gap_never_hands_out_a_used_number():
    """If .3 was named by hand after .0 and .1, the next is .4, not .2 or .3: a release
    has to sort after every release before it."""
    tags = ["sweet/2026.10.0", "sweet/2026.10.1", "sweet/2026.10.3"]
    assert next_release(tags, "2026.10") == "sweet/2026.10.4"


def test_surrounding_whitespace_in_the_tag_list_is_tolerated():
    assert next_release(["  sweet/2026.10.0  ", ""], "2026.10") == "sweet/2026.10.1"


@pytest.mark.parametrize("month", ["2026.13", "2026.00", "2026-10", "2026.1", "", "latest"])
def test_a_bad_month_is_refused(month):
    with pytest.raises(ValueError):
        next_release([], month)


@pytest.mark.parametrize(
    "name, ok",
    [
        ("sweet/2026.10.0", True),
        ("sweet/2026.10.12", True),
        ("sweet/2026.01.0", True),
        ("sweet/2026.13.0", False),  # matches the documented pattern; no such month
        ("sweet/2026.00.0", False),
        ("sweet/2026.10.01", False),
        ("wastemap/2026.10.07", False),
        ("2026.10.0", False),
    ],
)
def test_parse_accepts_exactly_the_release_names(name, ok):
    assert (next_release_mod.parse(name) is not None) is ok


def test_latest_is_numeric_and_ignores_everything_else():
    tags = OTHER_PROJECTS + [
        "sweet/2026.10.9",
        "sweet/2026.10.10",
        "sweet/2026.09.30",
        "sweet/2025.12.99",
        "sweet/test",
    ]
    assert latest_release(tags) == "sweet/2026.10.10"


def test_latest_of_no_releases_is_none():
    assert latest_release(OTHER_PROJECTS) is None
    assert latest_release([]) is None


# --- the command line the workflow actually runs -----------------------------------


def _run(*args, stdin=""):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_next_reads_tags_from_stdin():
    out = _run("2026.10", stdin="sweet/2026.10.0\nwastemap/2026.10.07\nsweet/2026.10.1\n")
    assert out.returncode == 0, out.stderr
    assert out.stdout == "sweet/2026.10.2\n"


def test_cli_next_defaults_to_the_current_utc_month():
    before = next_release_mod._current_month()
    out = _run(stdin="")
    after = next_release_mod._current_month()  # in case the month turned over mid-test
    assert out.returncode == 0, out.stderr
    assert out.stdout in {f"sweet/{before}.0\n", f"sweet/{after}.0\n"}


def test_cli_check_prints_the_trimmed_name_or_refuses():
    good = _run("--check", " sweet/2026.10.3 ")
    assert (good.returncode, good.stdout) == (0, "sweet/2026.10.3\n")

    bad = _run("--check", "sweet/2026.10.03")
    assert bad.returncode == 1
    assert bad.stdout == ""
    assert "not a release name" in bad.stderr


def test_cli_latest_prints_nothing_when_there_is_no_release():
    assert _run("--latest", stdin="wastemap/2026.10.07\n").stdout == ""
    assert _run("--latest", stdin="sweet/2026.09.0\nsweet/2026.10.0\n").stdout == "sweet/2026.10.0\n"


def test_cli_bad_month_exits_nonzero():
    out = _run("2026.13")
    assert out.returncode == 2
    assert out.stdout == ""
