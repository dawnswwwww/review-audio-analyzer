"""Terminology dictionary and candidate marking for transcript cleaning.

A terminology entry maps a canonical technical term to a set of known
ASR-misheard aliases.  The cleaner uses these entries to pre-mark candidates
before the LLM makes a context-aware decision.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from pypinyin import lazy_pinyin
except ImportError:  # pragma: no cover - pypinyin is a declared dependency

    def lazy_pinyin(text: str, **_kwargs: object) -> list[str]:  # type: ignore[misc]
        return list(text)


@dataclass(frozen=True)
class TermEntry:
    """A single terminology entry mapping a canonical term to known aliases."""

    correct: str
    aliases: tuple[str, ...]
    category: str | None = None

    def __init__(
        self, correct: str, aliases: tuple[str, ...] | list[str], category: str | None = None
    ) -> None:
        object.__setattr__(self, "correct", correct)
        object.__setattr__(self, "aliases", tuple(aliases))
        object.__setattr__(self, "category", category)


@dataclass(frozen=True)
class _InternalTermEntry:
    correct: str
    aliases: tuple[str, ...]
    category: str | None = None


@dataclass(frozen=True)
class CorrectionCandidate:
    """A candidate correction found in a single utterance."""

    original: str
    suggested: str
    utterance_index: int
    source: str
    confidence: float


class Terminology:
    """A collection of terminology entries used to pre-mark ASR errors."""

    def __init__(self, entries: list[TermEntry]) -> None:
        self._entries = list(entries)
        # Build an index of alias -> canonical form for fast exact matching.
        self._alias_to_correct: dict[str, str] = {}
        self._pinyin_to_entries: dict[str, list[TermEntry]] = {}
        for entry in self._entries:
            for alias in entry.aliases:
                self._alias_to_correct[alias.lower()] = entry.correct
            pinyin = "".join(lazy_pinyin(entry.correct)).lower()
            self._pinyin_to_entries.setdefault(pinyin, []).append(entry)

    @classmethod
    def load(cls, path: str | Path) -> Terminology:
        """Load terminology from a JSON file.

        Expected shape::

            {
              "terms": [
                {"correct": "Fiber", "aliases": ["Fibre", "Fiber节点"], "category": "React"},
                ...
              ]
            }
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [
            TermEntry(
                correct=e["correct"],
                aliases=tuple(a for a in e.get("aliases", []) if a),
                category=e.get("category"),
            )
            for e in data.get("terms", [])
            if e.get("correct")
        ]
        return cls(entries)

    @classmethod
    def empty(cls) -> Terminology:
        return cls([])

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def fuzzy_candidates(self, text: str, utterance_index: int) -> list[CorrectionCandidate]:
        """Return candidates based on pinyin/phonetic similarity.

        This is intentionally conservative: it only proposes candidates whose
        pinyin is close to a known term, leaving the final decision to the LLM.
        """
        text_pinyin = "".join(lazy_pinyin(text)).lower()
        if not text_pinyin:
            return []

        candidates: list[CorrectionCandidate] = []
        seen: set[str] = set()
        for entry in self._entries:
            correct_pinyin = "".join(lazy_pinyin(entry.correct)).lower()
            if not correct_pinyin:
                continue
            # Simple edit-distance ratio on pinyin strings.
            dist = _levenshtein(text_pinyin, correct_pinyin)
            ratio = dist / max(len(text_pinyin), len(correct_pinyin))
            # High phonetic similarity but not identical -> candidate.
            if 0.0 < ratio <= 0.35:
                key = (text, entry.correct)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    CorrectionCandidate(
                        original=text,
                        suggested=entry.correct,
                        utterance_index=utterance_index,
                        source="terminology_phonetic",
                        confidence=round(1.0 - ratio, 2),
                    )
                )
        return candidates

    def find_candidates(self, text: str, utterance_index: int) -> list[CorrectionCandidate]:
        """Find all terminology-based candidates in ``text``."""
        candidates: list[CorrectionCandidate] = []
        # 1. Exact substring search for Chinese aliases and phrases that may be
        #    embedded in larger tokens (e.g. "前段" inside "前段的").
        for alias, correct in self._alias_to_correct.items():
            if alias in text.lower():
                candidates.append(
                    CorrectionCandidate(
                        original=alias,
                        suggested=correct,
                        utterance_index=utterance_index,
                        source="terminology_exact",
                        confidence=0.9,
                    )
                )

        # 2. Token-based matching for English/alphanumeric terms.
        tokens = _tokenize(text)
        for token in tokens:
            if token.lower() in self._alias_to_correct:
                continue  # already caught by substring search
            candidates.extend(self.fuzzy_candidates(token, utterance_index))
        return candidates


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[一-鿿]+")


def _tokenize(text: str) -> list[str]:
    """Split text into candidate tokens for terminology matching."""
    return _TOKEN_RE.findall(text)


def _levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ach in enumerate(a, start=1):
        curr = [i]
        for j, bch in enumerate(b, start=1):
            cost = 0 if ach == bch else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def build_candidate_summary(candidates: list[CorrectionCandidate]) -> list[dict]:
    """Convert candidates to a JSON-serialisable summary for the LLM prompt."""
    by_original: dict[str, list[CorrectionCandidate]] = {}
    for c in candidates:
        by_original.setdefault(c.original, []).append(c)

    result: list[dict] = []
    for original, group in by_original.items():
        best = max(group, key=lambda c: c.confidence)
        result.append(
            {
                "original": original,
                "suggested": best.suggested,
                "source": best.source,
                "confidence": best.confidence,
            }
        )
    return sorted(result, key=lambda x: (-x["confidence"], x["original"]))
