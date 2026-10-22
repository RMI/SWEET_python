"""Reads `MODEL_DECISIONS.md`, SWEET's list of its own modelling decisions.

A decision is a choice SWEET makes that changes a number a tool built on it reports,
and that someone could reasonably have made differently: a fallback, a fixed value, a
simplification, or a choice between two methods. The markdown file is the list; this
module only reads it.

The list ships inside the package, beside this module, so an app reads the list of
the exact SWEET release it pins: `load()` with no argument. Climate TRACE and WasteMAP
keep lists of their own decisions, which link to entries here and rate how much each
moves their own numbers. SWEET's entries carry no impact rating for that reason.

Where the code applies a decision, a one-line comment names it, in the form
"decision: <id>" after a hash. `tests/test_model_decisions.py` fails when a comment
and the list disagree, in either direction.

Standard library only, so any caller can import it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DOC = Path(__file__).with_name("MODEL_DECISIONS.md")

KINDS = ("fallback", "fixed value", "simplification", "method choice")
RATINGS = ("high", "medium", "low")
SECTIONS = ("Inputs", "Diversion", "Decay model", "Outputs")
RETIRED_SECTION = "Retired"

#: The entry points a decision can apply on, as an entry's Paths field names them.
PATHS = {
    "city-model": "City.load_csv_new / load_andre_params: WasteMAP's cities table and city DST baseline",
    "city-dst-scenario": "City.implement_dst_changes_simple*: WasteMAP's city DST scenarios",
    "custom-location": "City.dst_baseline_blank: WasteMAP's Custom Location",
    "site-tool": "City.sdst_v1_5: WasteMAP's single-site tool",
    "ct-site": "City.site_only_estimate_trace: Climate TRACE's site estimates",
    "adst-site": "advanced_dst.run_advanced_dst: the single-site advanced DST",
    "adst-city": "advanced_dst_city.run_advanced_dst_city: WasteMAP's advanced DST",
}

#: Text fields capped at this many words. Longer reasoning goes in a linked doc.
MAX_WORDS = 40

#: The comment that marks where code applies a decision. One or more ids, comma-separated.
TAG = re.compile(r"#\s*decision:\s*([a-z0-9.\-]+(?:\s*,\s*[a-z0-9.\-]+)*)")

_SECTION_HEADING = re.compile(r"^## (.+?)\s*$")
_ENTRY_HEADING = re.compile(r"^### `([^`]+)`\s*$")
_FIELD = re.compile(r"^- \*\*([^*:]+):\*\*\s*(.*)$")
_OPTION = re.compile(r"^\s+\d+\.\s+`([^`]+)`:\s*(.*)$")
#: "  - `city-model`, `custom-location`: what those paths do"
_PATH_ITEM = re.compile(r"^\s+- ((?:`[^`]+`(?:,\s*)?)+):\s*(.*)$")
_CODE_PATH = re.compile(r"`([^`]+\.py)`(?:\s*\((\d+) places\))?")


@dataclass
class Decision:
    id: str
    section: str
    what: str
    fields: dict = field(default_factory=dict)
    options: list = field(default_factory=list)  # [(name, text), ...] best first
    paths: list = field(default_factory=list)  # [([path names], text), ...]
    line: int = 0

    @property
    def retired(self) -> bool:
        return self.section == RETIRED_SECTION

    @property
    def kind(self) -> str:
        return self.fields.get("Kind", "")

    @property
    def option_names(self) -> list:
        return [name for name, _text in self.options]

    @property
    def path_names(self) -> list:
        """Every path an item names, in order. A path named twice appears twice."""
        return [name for names, _text in self.paths for name in names]

    @property
    def code_places(self) -> dict:
        """{file: number of places in it that carry the comment}. One unless the field says
        "(N places)" after the file. Files are repo-relative: `SWEET_python/...`."""
        return {
            path: int(n) if n else 1
            for path, n in _CODE_PATH.findall(self.fields.get("Code", ""))
        }


def load(path: Path = DOC) -> list:
    """Every entry in the list, active and retired, in file order."""
    decisions = []
    section = None
    current = None
    in_list = None  # "options" or "paths" while reading a field's indented items
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m = _SECTION_HEADING.match(raw)
        if m:
            section, current, in_list = m.group(1), None, None
            continue
        m = _ENTRY_HEADING.match(raw)
        if m:
            current = Decision(id=m.group(1), section=section or "", what="", line=n)
            decisions.append(current)
            in_list = None
            continue
        if current is None:
            continue
        m = _FIELD.match(raw)
        if m:
            name, value = m.group(1).strip(), m.group(2).strip()
            current.fields[name] = value
            in_list = "options" if name.startswith("Options") else "paths" if name == "Paths" else None
            continue
        m = _OPTION.match(raw)
        if m and in_list == "options":
            current.options.append((m.group(1), m.group(2).strip()))
            continue
        m = _PATH_ITEM.match(raw)
        if m and in_list == "paths":
            current.paths.append((re.findall(r"`([^`]+)`", m.group(1)), m.group(2).strip()))
            continue
        if raw.strip() and not current.fields and not raw.startswith(("|", "#")):
            # The sentence under the heading, before the first field, is "what".
            current.what = f"{current.what} {raw.strip()}".strip()
    return decisions


def by_id(path: Path = DOC) -> dict:
    return {d.id: d for d in load(path)}


def tags_in(text: str) -> list:
    """Ids named by decision comments in one file's text."""
    ids = []
    for m in TAG.finditer(text):
        ids.extend(part.strip() for part in m.group(1).split(","))
    return ids
