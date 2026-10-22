"""`SWEET_python/MODEL_DECISIONS.md` and the decision comments in the code agree.

A list of assumptions is only useful while it matches the code, so the coupling is
mechanical. Each entry's `Code` field names the files that apply it, and each of those
files carries a comment naming the entry. Remove the comment, rename the entry, or tag a
new place without listing it, and this test fails. Counts are exact: an entry's Code
field says "(N places)" after a file that carries the comment more than once.

The list ships inside the package, so an app reads the list of the release it pins. Climate
TRACE's and WasteMAP's own lists link to these entries; their tests check the ids they link
to exist here.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from SWEET_python import model_decisions as md

REPO_ROOT = Path(md.__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "SWEET_python"
_SKIP_FILES = {Path(md.__file__).resolve()}
_REQUIRED_ACTIVE = ("Kind", "Why", "Paths", "Confidence", "Code", "Reviewed")
_REQUIRED_RETIRED = ("Kind", "Why", "Retired")


def _tags() -> dict:
    """{repo-relative path: [ids]} for every package file with a decision comment."""
    found = {}
    for path in PACKAGE.rglob("*.py"):
        if path.resolve() in _SKIP_FILES or "__pycache__" in path.parts:
            continue
        ids = md.tags_in(path.read_text(encoding="utf-8", errors="replace"))
        if ids:
            found[path.relative_to(REPO_ROOT).as_posix()] = ids
    return found


@pytest.fixture(scope="module")
def decisions():
    return md.load()


@pytest.fixture(scope="module")
def active(decisions):
    return [d for d in decisions if not d.retired]


def test_the_list_ships_with_the_package():
    """An app installs SWEET from git with pip, which copies only what MANIFEST.in names."""
    assert md.DOC.exists()
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "SWEET_python/MODEL_DECISIONS.md" in manifest


def test_ids_are_unique_and_well_formed(decisions):
    ids = [d.id for d in decisions]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"ids used twice: {dupes}. An id is never reused."
    bad = [i for i in ids if not re.fullmatch(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", i)]
    assert not bad, f"ids must be lower-case words joined by hyphens: {bad}"


def test_every_entry_sits_under_a_known_section(decisions):
    allowed = set(md.SECTIONS) | {md.RETIRED_SECTION}
    wrong = {d.id: d.section for d in decisions if d.section not in allowed}
    assert not wrong, f"entries under an unknown '## ' heading: {wrong}. Allowed: {sorted(allowed)}"


def test_every_entry_has_its_fields(decisions):
    problems = []
    for d in decisions:
        required = _REQUIRED_RETIRED if d.retired else _REQUIRED_ACTIVE
        problems += [f"{d.id}: no {f}" for f in required if f != "Paths" and not d.fields.get(f)]
        if not d.what:
            problems.append(f"{d.id}: no sentence under the heading saying what SWEET does")
        if d.kind not in md.KINDS:
            problems.append(f"{d.id}: Kind {d.kind!r} is not one of {md.KINDS}")
        if not d.retired and d.fields.get("Confidence") not in md.RATINGS:
            problems.append(f"{d.id}: Confidence {d.fields.get('Confidence')!r} is not one of {md.RATINGS}")
        if "Impact" in d.fields:
            problems.append(f"{d.id}: SWEET rates no impact; each app's own list does")
    assert not problems, "\n".join(problems)


def test_every_active_entry_names_the_paths_it_applies_on(active):
    problems = []
    for d in active:
        names = d.path_names
        if not names:
            problems.append(f"{d.id}: no Paths items (`- \\`path\\`: what it does there`)")
        unknown = sorted(set(names) - set(md.PATHS))
        if unknown:
            problems.append(f"{d.id}: unknown paths {unknown}; known: {sorted(md.PATHS)}")
        if len(set(names)) != len(names):
            problems.append(f"{d.id}: a path is listed twice")
    assert not problems, "\n".join(problems)


def test_fallbacks_list_their_options_and_nothing_else_does(decisions):
    problems = []
    for d in decisions:
        names = d.option_names
        if d.kind == "fallback" and not d.retired and len(names) < 2:
            problems.append(f"{d.id}: a fallback needs at least two options, best first")
        if d.kind != "fallback" and names:
            problems.append(f"{d.id}: only a fallback lists options")
        if len(set(names)) != len(names):
            problems.append(f"{d.id}: option names repeat: {names}")
        bad = [n for n in names if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", n)]
        if bad:
            problems.append(f"{d.id}: option names must be lower-case words joined by hyphens: {bad}")
    assert not problems, "\n".join(problems)


def test_text_stays_short(decisions):
    """Long reasoning goes in a linked doc. The entry says what and why in a sentence."""
    long = []
    for d in decisions:
        texts = {"what": d.what, "Why": d.fields.get("Why", ""), "Better": d.fields.get("Better", "")}
        texts.update({f"option {n}": t for n, t in d.options})
        texts.update({f"path {', '.join(n)}": t for n, t in d.paths})
        long += [
            f"{d.id} {name}: {len(t.split())} words"
            for name, t in texts.items()
            if len(t.split()) > md.MAX_WORDS
        ]
    assert not long, f"over {md.MAX_WORDS} words:\n" + "\n".join(long)


def test_reviewed_and_retired_dates_are_real_dates(decisions):
    bad = []
    for d in decisions:
        for f in ("Reviewed", "Retired"):
            value = d.fields.get(f)
            if not value:
                continue
            m = re.match(r"(\d{4}-\d{2}-\d{2})", value)
            if not m or dt.date.fromisoformat(m.group(1)) > dt.date.today():
                bad.append(f"{d.id} {f}: {value!r}")
    assert not bad, "dates must start YYYY-MM-DD and not be in the future: " + ", ".join(bad)


def test_every_listed_file_carries_the_comment(active):
    tags = _tags()
    wrong = [
        f"{d.id}: {p} has {tags.get(p, []).count(d.id)}, Code says {n}"
        for d in active
        for p, n in d.code_places.items()
        if tags.get(p, []).count(d.id) != n
    ]
    assert not wrong, (
        "an entry's Code field and the `# decision: <id>` comments disagree. A file with 0 has "
        "lost its comment, or does not exist. Add or remove a comment, or fix the count "
        "written as \"(N places)\":\n" + "\n".join(wrong)
    )


def test_every_comment_is_listed(decisions):
    by_id = {d.id: d for d in decisions}
    problems = []
    for path, ids in _tags().items():
        for i in ids:
            d = by_id.get(i)
            if d is None:
                problems.append(f"{path}: `{i}` is not in {md.DOC.name}")
            elif d.retired:
                problems.append(f"{path}: `{i}` is retired, so the code should not apply it")
            elif path not in d.code_places:
                problems.append(f"{path}: `{i}` is tagged here but its Code field does not list this file")
    assert not problems, "\n".join(problems)


def test_the_summary_table_matches_the_entries(active):
    """The table at the top is a second copy of three fields, so it is pinned."""
    text = md.DOC.read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| \[`([^`]+)`\]\(#[^)]*\) \| ([^|]+?) \| ([^|]+?) \| ([^|]+?) \|$",
        text,
        flags=re.M,
    )
    table = {r[0]: tuple(c.strip() for c in r[1:]) for r in rows}
    entries = {d.id: (d.section, d.kind, d.fields["Confidence"]) for d in active}
    assert table == entries, (
        "the summary table and the entries disagree.\n"
        f"only in the table: {sorted(set(table) - set(entries))}\n"
        f"only in entries: {sorted(set(entries) - set(table))}\n"
        f"different: {sorted(i for i in set(table) & set(entries) if table[i] != entries[i])}"
    )


def test_the_list_reads_from_the_installed_package():
    """What an app calls: no path, the copy beside the module."""
    assert {d.id for d in md.load()} == {d.id for d in md.load(md.DOC)}
    assert md.by_id()  # not empty
