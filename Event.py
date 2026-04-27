# Event.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from Signal import Hit


# ---------------- chamber container ----------------

@dataclass(frozen=True, slots=True)
class ChamberBlock:
    """
    One chamber's payload inside an Event.
    """
    chamber_id: int
    hits: Tuple[Hit, ...]  # immutable hits for this chamber

    @property
    def n_hits(self) -> int:
        return len(self.hits)


# Alias: chamber_id -> ChamberBlock
Chambers = Dict[int, ChamberBlock]


# ---------------- event container ----------------

@dataclass(frozen=True, slots=True)
class Event:
    event_id20: int
    rd_bank_sel: int
    trigger_count: int
    hit_count_expected: int

    # chamber_id -> ChamberBlock
    chambers: Chambers = field(default_factory=dict)