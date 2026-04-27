# prefilter_ml10.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

from Signal import Hit
from Event import Event
from geometry import Geometry

LEDGE_BITS = 17
LEDGE_MOD  = 1 << LEDGE_BITS


def dedup_hits_first_occurrence(hits: List[Hit]) -> List[Hit]:
    """
    Deduplicate hits by (tdcid, ch).

    Assumption:
      - Hits from the same tube arrive in time order within the event.

    Rule:
      - Keep the first hit for each (tdcid, ch)
      - Discard all subsequent hits from the same tube
      - Preserve original order
    """
    seen_tubes = set()
    kept_hits = []

    for hit in hits:
        tube_id = (int(hit.tdcid), int(hit.ch))
        if tube_id in seen_tubes:
            continue
        seen_tubes.add(tube_id)
        kept_hits.append(hit)

    return kept_hits




# ---------- 17-bit minimal-span t0 + span cut ----------
def choose_t0_and_check_span_17bit(hits: List[Hit], drift_time_max_ticks: int) -> Tuple[bool, int, int]:
    """
    Returns (ok, t0_raw_ledge, span_ticks)
    Uses minimal-span unwrapping to handle rollover robustly.
    """
    if not hits:
        return False, 0, 0

    ledges = sorted((int(h.ledge) & (LEDGE_MOD - 1)) for h in hits)
    n = len(ledges)
    if n == 1:
        return True, ledges[0], 0

    # gaps around ring
    gaps = [ledges[i+1] - ledges[i] for i in range(n-1)]
    gaps.append((ledges[0] + LEDGE_MOD) - ledges[-1])  # wrap gap

    k = max(range(n), key=lambda i: gaps[i])  # largest gap index
    start = (k + 1) % n

    # unwrap
    unwrapped = []
    last = None
    for i in range(n):
        v = ledges[(start + i) % n]
        if last is None:
            unwrapped.append(v)
            last = v
        else:
            if v < last:
                v += LEDGE_MOD
            unwrapped.append(v)
            last = v

    span = int(unwrapped[-1] - unwrapped[0])
    if span > int(drift_time_max_ticks):
        return False, 0, span

    t0_raw = int(unwrapped[0] & (LEDGE_MOD - 1))
    return True, t0_raw, span


# ---------- clustering (pairwise abs) for N<=10 ----------
def largest_cluster_indices(coords: List[Tuple[int, int]]) -> Set[int]:
    """
    coords: list of (local_layer, col) for hits in one ML
    adjacency: |dl|<=1 and |dc|<=1
    Returns indices of the largest connected component.
    """
    n = len(coords)
    if n == 0:
        return set()

    unvisited = set(range(n))
    best: Set[int] = set()

    while unvisited:
        seed = unvisited.pop()
        q = deque([seed])
        comp = {seed}

        while q:
            i = q.popleft()
            li, ci = coords[i]
            to_add = []
            for j in unvisited:
                lj, cj = coords[j]
                if abs(li - lj) <= 1 and abs(ci - cj) <= 1:
                    to_add.append(j)
            for j in to_add:
                unvisited.remove(j)
                q.append(j)
                comp.add(j)

        if len(comp) > len(best):
            best = comp

    return best


def split_hits_by_ml(geo: Geometry, hits: List[Hit]) -> Tuple[List[int], List[int], List[Tuple[int,int,int]]]:
    """
    Returns:
      idx_ml0, idx_ml1, info_per_hit where info=(ml, local_layer, col)
    """
    idx_ml0: List[int] = []
    idx_ml1: List[int] = []
    info: List[Tuple[int, int, int]] = []

    for i, h in enumerate(hits):
        _x, _y, layer, col = geo.wire_center_from_hit(int(h.tdcid), int(h.ch))
        ml = geo.multilayer_from_layer(layer)
        local_layer = int(layer) - int(ml) * int(geo.MAX_TDC_LAYER)
        info.append((int(ml), int(local_layer), int(col)))
        if ml == 0:
            idx_ml0.append(i)
        elif ml == 1:
            idx_ml1.append(i)

    return idx_ml0, idx_ml1, info


@dataclass(frozen=True, slots=True)
class PrefilterResult:
    ok: bool
    reason: str
    hits_kept: List[Hit]
    t0_raw_ledge: int
    span_ticks: int
    n_in: int
    n_keep: int
    n_ml0: int
    n_ml1: int


def prefilter_event_ml10(
    ev: Event,
    geo: Geometry,
    *,
    min_hits: int,
    max_hits: int,
    drift_time_max_ns: float,
    tick_ns: float,
    max_hits_per_ml: int = 10,
) -> PrefilterResult:
    """
    Your requested prefilter with an extra rule:
      discard event if hits in ML0 or ML1 exceed max_hits_per_ml (default 10).
    """
    # 0) dedup by (tdcid,ch) keep earliest
    hits0 = dedup_hits_keep_earliest_stable(ev.hits)

    n = len(hits0)
    if n < int(min_hits) or n > int(max_hits):
        return PrefilterResult(False, "hit_count", [], 0, 0, n, 0, 0, 0)

    # 1) split by ML and hard-discard if >10 in any ML
    idx0, idx1, info = split_hits_by_ml(geo, hits0)
    n0, n1 = len(idx0), len(idx1)
    if n0 > int(max_hits_per_ml) or n1 > int(max_hits_per_ml):
        return PrefilterResult(False, "too_many_hits_per_ml", [], 0, 0, n, 0, n0, n1)

    # 2) ledge span < drift_time_max, choose t0
    drift_time_max_ticks = int(round(float(drift_time_max_ns) / float(tick_ns)))
    ok_t0, t0_raw, span = choose_t0_and_check_span_17bit(hits0, drift_time_max_ticks)
    if not ok_t0:
        return PrefilterResult(False, "ledge_span", [], 0, span, n, 0, n0, n1)

    # 3) cluster per ML and keep largest cluster in each ML
    keep_global: Set[int] = set()

    if idx0:
        coords0 = [(info[i][1], info[i][2]) for i in idx0]  # (local_layer, col)
        keep0_local = largest_cluster_indices(coords0)
        keep_global.update(idx0[j] for j in keep0_local)

    if idx1:
        coords1 = [(info[i][1], info[i][2]) for i in idx1]
        keep1_local = largest_cluster_indices(coords1)
        keep_global.update(idx1[j] for j in keep1_local)

    hits_kept = [hits0[i] for i in sorted(keep_global)]
    if len(hits_kept) < int(min_hits):
        return PrefilterResult(False, "cluster_too_small", [], t0_raw, span, n, len(hits_kept), n0, n1)

    return PrefilterResult(True, "ok", hits_kept, t0_raw, span, n, len(hits_kept), n0, n1)
