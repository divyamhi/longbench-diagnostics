"""
parse_response.py
-----------------
Parses raw model output into a single answer letter (A, B, C, or D).

This module is intentionally dependency-free. It performs purely
lexical extraction using regex and has no imports beyond the standard
library. It is used by all model backends and prompt strategies.

Design rules:
  - Never raises an exception; always returns str | None.
  - Returns None on any ambiguity (zero matches OR multiple distinct matches).
  - Treats the raw string case-insensitively before matching.
"""

import re


def extract_choice(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    if not raw.strip():
        return None
    uppercased = raw.upper()
    matches = re.findall(r'\b([ABCD])\b', uppercased)
    unique_matches = set(matches)
    if len(unique_matches) == 1:
        return unique_matches.pop()
    return None
