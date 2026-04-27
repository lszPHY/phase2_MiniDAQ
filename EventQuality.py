from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Any
import numpy as np

from Signal import Hit


# ============================================================
# Slew correction (original class preserved)
# ============================================================

class TimeSlewASD2:
    """
    C++-equivalent slew correction, ASD2 always.
    Input: Hit.width raw count (0..255)
    Output: correction in ns
    """

    LEDGE_LSB_NS = 0.78125
    LEDGE_ROLLOVER = 1 << 17  # 17-bit counter

    WIDTH_LSB_NS = 2.0 * 0.78125  # 1.5625 ns
    SLEW_TABLE_N = 400            # C++ table size

    def __init__(self, coefficient: float = 35.59, scale: float = 0.0163):
        self.scale = float(scale)
        self.coeff = float(coefficient) * 1.33  # ASD2 multiplier
        self.table = self._build_table()

    def _build_table(self) -> np.ndarray:
        i = np.arange(self.SLEW_TABLE_N, dtype=np.float64)
        charge = 0.8 * i  # WidthToCharge for ASD2
        return self.coeff / np.exp(self.scale * charge)  # ns

    def corr_ns(self, width_cnt: int) -> float:
        w_cnt = int(width_cnt) & 0xFF
        w_ns = w_cnt * self.WIDTH_LSB_NS
        wbin = int(w_ns)  # truncate like C++ static_cast<int>
        if 0 <= wbin < self.SLEW_TABLE_N:
            return float(self.table[wbin])
        return 0.0


# ============================================================
# Event Quality Checker
# ============================================================

class EventQualityChecker:

    LEDGE_LSB_NS = 0.78125
    LEDGE_ROLLOVER = 1 << 17

    @dataclass(frozen=True, slots=True)
    class TimeWindowResult:
        t_corr_ns: np.ndarray
        keep_mask: np.ndarray
        valid: bool

    @dataclass(frozen=True, slots=True)
    class Result:
        time_by_chamber: Dict[int, "EventQualityChecker.TimeWindowResult"]
        valid_by_chamber: Dict[int, bool]

    def __init__(
        self,
        *,
        min_window_ns: float = 10.0,
        max_window_ns: float = 250.0,

        # hit count constraints
        n_layers: int = 8,
        min_total_hits: int = 0,
        max_total_hits: int = 10000,
        min_layer_hit: int = 0,
        max_hit_per_layer: int = 10000,
        max_unique_col_per_layer: int = 10000,
        max_adjacent_layer_x_jump_mm: float = float("inf"),
        max_same_layer_x_span_mm: float = float("inf"),

        # NEW: island (multi-cluster) constraints, checked per multilayer (ML)
        max_col_gap: int = 2,          # gap > this starts a new island
        max_col_clusters: int = 1,     # require <= this many islands per ML

        # flags controlling whether checks affect validity
        checkTime: bool = True,
        checkHitcount: bool = True,

        slew: Optional[TimeSlewASD2] = None,
    ):

        self.min_window_ns = float(min_window_ns)
        self.max_window_ns = float(max_window_ns)

        self.n_layers = int(n_layers)

        self.min_total_hits = int(min_total_hits)
        self.max_total_hits = int(max_total_hits)

        self.min_layer_hit = int(min_layer_hit)
        self.max_hit_per_layer = int(max_hit_per_layer)
        self.max_unique_col_per_layer = int(max_unique_col_per_layer)
        self.max_adjacent_layer_x_jump_mm = float(max_adjacent_layer_x_jump_mm)
        self.max_same_layer_x_span_mm = float(max_same_layer_x_span_mm)

        # NEW: island parameters
        self.max_col_gap = int(max_col_gap)
        self.max_col_clusters = int(max_col_clusters)

        # control whether each check affects the final validity flag
        self.checkTime = bool(checkTime)
        self.checkHitcount = bool(checkHitcount)

        self.slew = slew if slew is not None else TimeSlewASD2()

    # --------------------------------------------------------
    # Exact translation of your C++ rollover subtraction
    # --------------------------------------------------------

    def _rollover_diff(self, a: int, b: int) -> int:
        """
        Exact translation of your C++:
          bindiff = a-b;
          if (bindiff > rollover/2) bindiff -= rollover;
          else if (bindiff < -rollover/2) bindiff += rollover;
        """
        d = int(a) - int(b)
        half = self.LEDGE_ROLLOVER // 2

        if d > half:
            d -= self.LEDGE_ROLLOVER
        elif d < -half:
            d += self.LEDGE_ROLLOVER

        return int(d)

    # --------------------------------------------------------
    # Time window check
    # --------------------------------------------------------

    def _time_window_check(self, hits: Iterable[Hit]) -> TimeWindowResult:
        """
        Your requested algorithm:

        1) Do NOT sort hits. Use the FIRST hit ledge as reference.
        2) dt_counts = rollover_diff(ledge_i, ledge_0)
        3) Convert to raw time: t_raw_ns = dt_counts * 0.78125 ns
        4) Apply time slew: t_corr_ns = t_raw_ns - SlewCorrection(width_cnt)
        5) Shift so the smallest corrected time = 0.
        6) If min_window_ns <= max(t_corr_ns) <= max_window_ns ? valid.
           - If valid ? keep all hits.
           - If not valid ? only keep hits inside window.
        """

        hits = list(hits)
        n = len(hits)

        if n == 0:
            return self.TimeWindowResult(
                t_corr_ns=np.zeros(0),
                keep_mask=np.zeros(0, dtype=bool),
                valid=False,
            )

        # 1) first hit as reference (NO sorting)
        ledge0 = int(hits[0].ledge) & (self.LEDGE_ROLLOVER - 1)

        # 2) rollover-corrected relative counts vs first hit
        dt_counts = np.array(
            [
                self._rollover_diff(
                    int(h.ledge) & (self.LEDGE_ROLLOVER - 1),
                    ledge0,
                )
                for h in hits
            ],
            dtype=np.int64,
        )

        # 3) raw timing in ns
        t_raw = dt_counts.astype(np.float64) * self.LEDGE_LSB_NS

        # 4) slew correction per hit
        slew_corr = np.array(
            [self.slew.corr_ns(int(h.width)) for h in hits],
            dtype=np.float64,
        )

        t_corr = t_raw - slew_corr

        # 5) shift so minimum = 0
        t_corr = t_corr - np.min(t_corr)

        span = float(np.max(t_corr))

        # validity condition
        valid = self.min_window_ns <= span <= self.max_window_ns

        # 6) keep logic
        if valid:
            keep = np.ones(n, dtype=bool)
        else:
            keep = (t_corr >= self.min_window_ns) & (t_corr <= self.max_window_ns)

        return self.TimeWindowResult(
            t_corr_ns=t_corr,
            keep_mask=keep,
            valid=bool(valid),
        )

    # --------------------------------------------------------
    # Island check (per multilayer)
    # --------------------------------------------------------

    def _ml_from_hit(self, h: Hit) -> int:
        """
        Decide ML for a hit.

        Priority:
          1) use h.ml if present
          2) derive from layer (default: 8 layers -> 4 layers per ML)
        """
        ml = getattr(h, "ml", None)
        if ml is not None:
            try:
                return int(ml)
            except Exception:
                pass

        layer = int(getattr(h, "layer", -1))
        if layer < 0:
            return -1

        layers_per_ml = max(1, self.n_layers // 2)  # 8 -> 4
        return int(layer // layers_per_ml)

    def _count_col_clusters(self, cols_sorted_unique: list[int]) -> int:
        """
        Count islands in sorted unique column list.
        A new cluster starts if gap > max_col_gap.
        """
        if not cols_sorted_unique:
            return 0

        clusters = 1
        gap = int(self.max_col_gap)

        for a, b in zip(cols_sorted_unique, cols_sorted_unique[1:]):
            if (b - a) > gap:
                clusters += 1
        return clusters

    def _island_check_per_ml(self, hits: list[Hit]) -> bool:
        """
        Reject events with multiple separated hit "islands" in column,
        checked separately in ML0 and ML1.

        Rule (per ML):
          clusters_in_col <= max_col_clusters
        """
        max_clusters = int(self.max_col_clusters)
        if max_clusters < 1:
            max_clusters = 1

        cols_by_ml: dict[int, set[int]] = {0: set(), 1: set()}

        for h in hits:
            ml = self._ml_from_hit(h)
            if ml not in (0, 1):
                continue

            col = int(getattr(h, "col", -1))
            if col >= 0:
                cols_by_ml[ml].add(col)

        for ml in (0, 1):
            cols = sorted(cols_by_ml[ml])
            clusters = self._count_col_clusters(cols)
            if clusters > max_clusters:
                return False

        return True

    # --------------------------------------------------------
    # Chamber hit count check
    # --------------------------------------------------------

    def _chamber_hit_check(self, hits: Iterable[Hit]) -> bool:
        """
        Checks:

          min_total_hits <= total_hits <= max_total_hits
          layer_hit >= min_layer_hit
          max(hit_per_layer) <= max_hit_per_layer
          max(unique tube columns per layer) <= max_unique_col_per_layer
          adjacent occupied layers do not jump too far in x
          same-layer x span is not too wide

        PLUS (NEW):

          island check per multilayer (ML):
            clusters_in_col_per_ML <= max_col_clusters
        """

        hits = list(hits)
        total_hits = len(hits)

        if not (self.min_total_hits <= total_hits <= self.max_total_hits):
            return False

        nL = max(1, int(self.n_layers))
        hits_per_layer = [0] * nL
        cols_per_layer: list[set[int]] = [set() for _ in range(nL)]
        xs_per_layer: list[list[float]] = [[] for _ in range(nL)]

        for h in hits:
            layer = int(getattr(h, "layer", -1))
            col = int(getattr(h, "col", -1))
            x = float(getattr(h, "x", np.nan))
            y = float(getattr(h, "y", np.nan))

            if not (0 <= layer < nL):
                return False
            if col < 0:
                return False
            if not (np.isfinite(x) and np.isfinite(y) and x >= 0.0 and y >= 0.0):
                return False

            hits_per_layer[layer] += 1
            cols_per_layer[layer].add(col)
            xs_per_layer[layer].append(x)

        # number of layers with >=1 hit
        layer_hit = sum(1 for c in hits_per_layer if c > 0)

        if layer_hit < self.min_layer_hit:
            return False

        # maximum hits in any layer
        if hits_per_layer and max(hits_per_layer) > self.max_hit_per_layer:
            return False

        max_unique_col = int(self.max_unique_col_per_layer)
        if max_unique_col >= 0:
            for cols in cols_per_layer:
                if len(cols) > max_unique_col:
                    return False

        max_same_span = float(self.max_same_layer_x_span_mm)
        if np.isfinite(max_same_span):
            for xs in xs_per_layer:
                if len(xs) >= 2 and (max(xs) - min(xs)) > max_same_span:
                    return False

        max_adj_jump = float(self.max_adjacent_layer_x_jump_mm)
        if np.isfinite(max_adj_jump):
            layer_x_center = []
            for layer, xs in enumerate(xs_per_layer):
                if xs:
                    layer_x_center.append((layer, float(np.mean(xs))))
            for (layer_a, xa), (layer_b, xb) in zip(layer_x_center, layer_x_center[1:]):
                if (layer_b - layer_a) == 1 and abs(xb - xa) > max_adj_jump:
                    return False

        # NEW: reject multi-island events (per ML)
        if not self._island_check_per_ml(hits):
            return False

        return True

    # --------------------------------------------------------
    # Main interface
    # --------------------------------------------------------

    def compute_for_event(self, ev: Any) -> Result:

        chambers = getattr(ev, "chambers", None)

        if not isinstance(chambers, dict):
            return self.Result({}, {})

        time_by_ch = {}
        valid_by_ch = {}

        for cid, block in chambers.items():

            hits = getattr(block, "hits", ())

            # always compute time correction (for drift circles etc.)
            tw = self._time_window_check(hits)

            # default validity
            valid = True

            # time validity only applied if enabled
            if self.checkTime:
                valid &= tw.valid

            # hitcount validity only applied if enabled
            if self.checkHitcount:
                valid &= self._chamber_hit_check(hits)

            time_by_ch[cid] = tw
            valid_by_ch[cid] = bool(valid)

        return self.Result(
            time_by_chamber=time_by_ch,
            valid_by_chamber=valid_by_ch,
        )
