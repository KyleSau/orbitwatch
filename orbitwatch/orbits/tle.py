"""Two-line element sets (TLEs): parsing and validation.

A TLE is a fixed-width text format. Line 1 holds the catalog number and the epoch
(the instant the elements describe); line 2 holds the mean orbital elements.
The last digit of each line is a mod-10 checksum, which we verify so a truncated
or corrupted download fails loudly instead of producing a wrong orbit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


class TleError(ValueError):
    """Raised for malformed TLE text."""


def tle_checksum(line: str) -> int:
    """Mod-10 checksum over the first 68 columns: digits count as themselves, '-' as 1."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


@dataclass(frozen=True)
class Tle:
    name: str
    line1: str
    line2: str

    @property
    def norad_id(self) -> int:
        return int(self.line1[2:7])

    @property
    def epoch(self) -> datetime:
        """Epoch from line 1 columns 19-32: two-digit year + fractional day of year."""
        yy = int(self.line1[18:20])
        day_of_year = float(self.line1[20:32])
        year = 2000 + yy if yy < 57 else 1900 + yy  # NORAD convention: 57-99 => 1900s
        return datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=day_of_year - 1)

    def age_days(self, now: datetime) -> float:
        return (now - self.epoch).total_seconds() / 86400.0

    def to_text(self) -> str:
        return f"{self.name}\n{self.line1}\n{self.line2}\n"


def _validate_line(line: str, number: str) -> None:
    if len(line) != 69 or not line.startswith(number + " "):
        raise TleError(f"Line {number} must be 69 characters starting with '{number} ': {line!r}")
    if not line[68].isdigit() or tle_checksum(line) != int(line[68]):
        raise TleError(f"Checksum mismatch on line {number}: {line!r}")


def parse_tle_text(text: str) -> list[Tle]:
    """Parse 3-line (name + 2 lines) or bare 2-line TLE text into Tle objects."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    tles: list[Tle] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name, l1, l2 = "", lines[i], lines[i + 1]
            i += 2
        elif i + 2 < len(lines):
            name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
            i += 3
        else:
            raise TleError(f"Incomplete TLE record near: {lines[i]!r}")
        _validate_line(l1, "1")
        _validate_line(l2, "2")
        if l1[2:7] != l2[2:7]:
            raise TleError("Catalog numbers on line 1 and line 2 differ.")
        tles.append(Tle(name=name or f"NORAD {int(l1[2:7])}", line1=l1, line2=l2))
    return tles
