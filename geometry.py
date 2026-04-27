# geometry.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class Geometry:
    # =========================
    # settings
    # =========================
    chamberType: str = "A"   # "A" or "C"
    flipTDCs: int = 1
    tdcColByTubeNo: int = 1

    # =========================
    # dimensions
    # =========================
    MAX_TDC: int = 40
    MAX_TDC_CHANNEL: int = 24
    MAX_TUBE_LAYER: int = 8
    MAX_TUBE_COLUMN: int = 60
    MAX_TDC_LAYER: int = 4

    ML_distance: float = 114.5695
    tube_length: float = 1.6715
    layer_distance: float = 13.0769836
    column_distance: float = 15.1
    radius: float = 7.5

    min_drift_dist: float = 0.0
    max_drift_dist: float = 7.1

    # =========================
    # assignment (persisted)
    # =========================
    slots_per_ml: int = 10
    ml0: List[Tuple[int, int]] = field(default_factory=list)  # (tdc_id, ncol)
    ml1: List[Tuple[int, int]] = field(default_factory=list)

    # =========================
    # auto-load
    # =========================
    geo_file: Optional[str] = None

    # =========================
    # runtime identity (NOT persisted)
    # =========================
    chamber_id: int = field(default=-1, repr=False, compare=False, metadata={"persist": False})

    # =========================
    # internal flags (NOT persisted)
    # =========================
    _skip_autoload: bool = field(default=False, repr=False, compare=False, metadata={"persist": False})

    # =========================
    # runtime arrays
    # =========================
    isActiveTDC: List[int] = field(default_factory=list)      # size MAX_TDC
    TDC_ML: List[int] = field(default_factory=list)           # size MAX_TDC
    TDC_COL: List[int] = field(default_factory=list)          # size MAX_TDC
    hit_layer_map: List[int] = field(default_factory=list)    # size MAX_TDC_CHANNEL
    hit_column_map: List[int] = field(default_factory=list)   # size MAX_TDC_CHANNEL
    tdc_map: List[Tuple[int, int, int]] = field(default_factory=list)  # (tdc, ml, colstart)

    # =============================================================================
    # init hook
    # =============================================================================

    def __post_init__(self) -> None:
        """
        Autoload policy:
          - If geo_file exists and _skip_autoload is False:
              load file into a temporary Geometry, then copy into self.
              Keep chamber_id from caller (not persisted).
          - If file missing / load fails:
              use init values.
        """
        # autoload first (so we don't build maps from wrong defaults)
        if (not self._skip_autoload) and self.geo_file:
            path = Path(self.geo_file)
            if path.is_file():
                try:
                    loaded = type(self).load(str(path), apply_map=True)  # loaded has its own post_init
                    keep_chamber_id = int(self.chamber_id)
                    self._copy_from(loaded)
                    self.chamber_id = keep_chamber_id
                    self.geo_file = str(path)
                    print(f"[Geometry] Loaded geometry from '{path}' (chamber_id={self.chamber_id})")
                    return
                except Exception as e:
                    print(f"[Geometry] Failed to load geometry from '{path}': {e} (using init values)")
            else:
                print(f"[Geometry] Geometry file not found: '{path}' (using init values)")

        # normal init path (no file)
        self._alloc()
        self.reset_tube_layout()
        self.set_assignment(self.slots_per_ml, self.ml0, self.ml1, apply_map=True)

    def _copy_from(self, other: "Geometry") -> None:
        """
        Copy ALL geometry config + runtime arrays from 'other' into self.
        Does NOT copy chamber_id (caller-owned identity).
        """
        # persisted config / settings
        self.chamberType = other.chamberType
        self.flipTDCs = int(other.flipTDCs)
        self.tdcColByTubeNo = int(other.tdcColByTubeNo)

        self.MAX_TDC = int(other.MAX_TDC)
        self.MAX_TDC_CHANNEL = int(other.MAX_TDC_CHANNEL)
        self.MAX_TUBE_LAYER = int(other.MAX_TUBE_LAYER)
        self.MAX_TUBE_COLUMN = int(other.MAX_TUBE_COLUMN)
        self.MAX_TDC_LAYER = int(other.MAX_TDC_LAYER)

        self.ML_distance = float(other.ML_distance)
        self.tube_length = float(other.tube_length)
        self.layer_distance = float(other.layer_distance)
        self.column_distance = float(other.column_distance)
        self.radius = float(other.radius)

        self.min_drift_dist = float(other.min_drift_dist)
        self.max_drift_dist = float(other.max_drift_dist)

        # assignment
        self.slots_per_ml = int(other.slots_per_ml)
        self.ml0 = list(other.ml0)
        self.ml1 = list(other.ml1)

        # runtime arrays / maps
        self.isActiveTDC = list(other.isActiveTDC)
        self.TDC_ML = list(other.TDC_ML)
        self.TDC_COL = list(other.TDC_COL)
        self.hit_layer_map = list(other.hit_layer_map)
        self.hit_column_map = list(other.hit_column_map)
        self.tdc_map = list(other.tdc_map)

    def _alloc(self) -> None:
        self.isActiveTDC = [0] * int(self.MAX_TDC)
        self.TDC_ML = [0] * int(self.MAX_TDC)
        self.TDC_COL = [0] * int(self.MAX_TDC)
        self.hit_layer_map = [0] * int(self.MAX_TDC_CHANNEL)
        self.hit_column_map = [0] * int(self.MAX_TDC_CHANNEL)

    # =============================================================================
    # Geometry mapping
    # =============================================================================

    def reset_tube_layout(self) -> None:
        for i in range(int(self.MAX_TDC_CHANNEL)):
            self.hit_layer_map[i] = (i + 2) % 4
            self.hit_column_map[i] = i // 4

    def configure_tdc_map(
        self,
        active_tdcs: List[int],
        tdc_multilayer: List[int],
        tdc_colstart: List[int],
        strict_duplicates: bool = False,
    ) -> None:
        if not (len(active_tdcs) == len(tdc_multilayer) == len(tdc_colstart)):
            raise ValueError("ActiveTDCs/TDCMultilayer/TDCColumn length mismatch")

        self.tdc_map = []
        self.isActiveTDC = [0] * int(self.MAX_TDC)
        self.TDC_ML = [0] * int(self.MAX_TDC)
        self.TDC_COL = [0] * int(self.MAX_TDC)

        seen = set()
        for tdc, ml, colstart in zip(active_tdcs, tdc_multilayer, tdc_colstart):
            tdc = int(tdc)
            ml = int(ml)
            colstart = int(colstart)

            if not (0 <= tdc < int(self.MAX_TDC)):
                continue
            if ml not in (0, 1):
                continue
            if strict_duplicates and tdc in seen:
                raise ValueError(f"Duplicate tdc {tdc}")
            seen.add(tdc)

            self.tdc_map.append((tdc, ml, colstart))
            self.isActiveTDC[tdc] = 1
            self.TDC_ML[tdc] = ml
            self.TDC_COL[tdc] = colstart

    def multilayer_from_layer(self, layer: int) -> int:
        return int(layer) // int(self.MAX_TDC_LAYER)

    def get_hit_layer_column(self, tdc_id: int, channel_id: int) -> Tuple[int, int]:
        if not (0 <= int(tdc_id) < int(self.MAX_TDC)):
            raise ValueError(f"tdc_id out of range: {tdc_id}")
        if not (0 <= int(channel_id) < int(self.MAX_TDC_CHANNEL)):
            raise ValueError(f"channel_id out of range: {channel_id}")
        if int(self.isActiveTDC[int(tdc_id)]) == 0:
            raise ValueError(f"tdc_id not active: {tdc_id}")

        hit_column = int(self.TDC_COL[int(tdc_id)]) + int(self.hit_column_map[int(channel_id)])
        hit_layer = int(self.MAX_TDC_LAYER) * int(self.TDC_ML[int(tdc_id)]) + int(self.hit_layer_map[int(channel_id)])
        return hit_layer, hit_column

    def get_hit_xy(self, hitL: int, hitC: int) -> Tuple[float, float]:
        if hitL < 0 or hitC < 0 or hitL >= int(self.MAX_TUBE_LAYER) or hitC >= int(self.MAX_TUBE_COLUMN):
            return -1.0, -1.0

        x = float(self.radius + hitC * self.column_distance + (hitL % 2) * (self.column_distance / 2.0))
        y = float(
            self.radius
            + hitL * self.layer_distance
            + (self.ML_distance - int(self.MAX_TDC_LAYER) * self.layer_distance) * self.multilayer_from_layer(hitL)
        )
        return x, y

    def wire_center_from_hit(self, tdc_id: int, channel_id: int) -> Tuple[float, float, int, int]:
        layer, col = self.get_hit_layer_column(tdc_id, channel_id)
        x, y = self.get_hit_xy(layer, col)
        return x, y, layer, col

    @staticmethod
    def channel_id_from_local(local_layer_0to3: int, local_col_0to5: int) -> int:
        ll = int(local_layer_0to3) & 3
        cc = int(local_col_0to5)
        if cc < 0:
            return -1
        k = (ll - 2) % 4
        return 4 * cc + k

    # =============================================================================
    # Assignment helpers
    # =============================================================================

    def _pad_and_clamp_slots(self, slots_per_ml: int, vals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        slots_per_ml = int(slots_per_ml)
        out = list(vals[:slots_per_ml])
        while len(out) < slots_per_ml:
            out.append((-1, 6))

        max_cols = int(self.MAX_TUBE_COLUMN)
        acc = 0
        out2: List[Tuple[int, int]] = []
        for tdc_id, ncol in out:
            tdc_id = int(tdc_id)
            ncol = max(0, int(ncol))
            if acc >= max_cols:
                ncol = 0
            else:
                ncol = min(ncol, max_cols - acc)
            out2.append((tdc_id, ncol))
            acc += ncol
        return out2

    def set_assignment(
        self,
        slots_per_ml: int,
        ml0: List[Tuple[int, int]],
        ml1: List[Tuple[int, int]],
        *,
        apply_map: bool = False,
        strict_duplicates: bool = False,
    ) -> None:
        self.slots_per_ml = int(slots_per_ml)
        self.ml0 = self._pad_and_clamp_slots(self.slots_per_ml, ml0)
        self.ml1 = self._pad_and_clamp_slots(self.slots_per_ml, ml1)

        if apply_map:
            active: List[int] = []
            mls: List[int] = []
            cols: List[int] = []

            acc = 0
            for tdc_id, ncol in self.ml0[: self.slots_per_ml]:
                if int(tdc_id) >= 0 and int(ncol) > 0:
                    active.append(int(tdc_id))
                    mls.append(0)
                    cols.append(int(acc))
                acc += max(0, int(ncol))

            acc = 0
            for tdc_id, ncol in self.ml1[: self.slots_per_ml]:
                if int(tdc_id) >= 0 and int(ncol) > 0:
                    active.append(int(tdc_id))
                    mls.append(1)
                    cols.append(int(acc))
                acc += max(0, int(ncol))

            self.configure_tdc_map(active, mls, cols, strict_duplicates=strict_duplicates)

    # =============================================================================
    # Serialization / loading
    # =============================================================================

    def to_text(self) -> str:
        g = self
        slots_per_ml = int(self.slots_per_ml)
        ml0p = self._pad_and_clamp_slots(slots_per_ml, self.ml0)
        ml1p = self._pad_and_clamp_slots(slots_per_ml, self.ml1)

        lines: List[str] = []
        lines.append("# geometry assignment")
        lines.append("")
        lines.append("chamberType " + str(g.chamberType))
        lines.append(f"flipTDCs {int(g.flipTDCs)}")
        lines.append(f"tdcColByTubeNo {int(g.tdcColByTubeNo)}")
        lines.append("")
        lines.append(f"MAX_TDC {int(g.MAX_TDC)}")
        lines.append(f"MAX_TDC_CHANNEL {int(g.MAX_TDC_CHANNEL)}")
        lines.append(f"MAX_TUBE_LAYER {int(g.MAX_TUBE_LAYER)}")
        lines.append(f"MAX_TUBE_COLUMN {int(g.MAX_TUBE_COLUMN)}")
        lines.append(f"MAX_TDC_LAYER {int(g.MAX_TDC_LAYER)}")
        lines.append("")
        lines.append(f"ML_distance {float(g.ML_distance)}")
        lines.append(f"tube_length {float(g.tube_length)}")
        lines.append(f"layer_distance {float(g.layer_distance)}")
        lines.append(f"column_distance {float(g.column_distance)}")
        lines.append(f"radius {float(g.radius)}")
        lines.append(f"min_drift_dist {float(g.min_drift_dist)}")
        lines.append(f"max_drift_dist {float(g.max_drift_dist)}")
        lines.append("")
        lines.append(f"slots_per_ml {slots_per_ml}")
        lines.append("")
        lines.append("ML0")
        for i, (tdc_id, ncol) in enumerate(ml0p[:slots_per_ml]):
            lines.append(f"{i} {int(tdc_id)} {int(ncol)}")
        lines.append("")
        lines.append("ML1")
        for i, (tdc_id, ncol) in enumerate(ml1p[:slots_per_ml]):
            lines.append(f"{i} {int(tdc_id)} {int(ncol)}")
        lines.append("")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_text())

    @classmethod
    def load(cls, path: str, *, apply_map: bool = False) -> "Geometry":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_text(f.read(), apply_map=apply_map)

    @classmethod
    def from_text(cls, text: str, *, apply_map: bool = False) -> "Geometry":
        gdict: Dict[str, Any] = {}
        slots_per_ml = 10
        ml0: List[Tuple[int, int]] = []
        ml1: List[Tuple[int, int]] = []
        cur_ml: Optional[int] = None

        allowed_geo_keys = {
            k for k, f in cls.__dataclass_fields__.items()
            if (not k.startswith("_")) and f.metadata.get("persist", True)
        }

        lines: List[str] = []
        for raw in text.splitlines():
            s = raw.strip()
            s = s.split("#", 1)[0].strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)

        for s in lines:
            up = s.upper()
            if up == "ML0":
                cur_ml = 0
                continue
            if up == "ML1":
                cur_ml = 1
                continue

            parts = s.split()

            if cur_ml in (0, 1) and len(parts) >= 3:
                try:
                    _slot = int(parts[0])
                    tdc_id = int(parts[1])
                    ncol = max(0, int(parts[2]))
                except ValueError:
                    continue
                (ml0 if cur_ml == 0 else ml1).append((tdc_id, ncol))
                continue

            if len(parts) == 2:
                k, v = parts[0], parts[1]

                if k == "slots_per_ml":
                    try:
                        slots_per_ml = int(v)
                    except ValueError:
                        pass
                    continue

                if k not in allowed_geo_keys:
                    continue

                if k == "chamberType":
                    gdict[k] = str(v)
                    continue

                if k.startswith("MAX_") or k in ("flipTDCs", "tdcColByTubeNo"):
                    try:
                        gdict[k] = int(v)
                    except ValueError:
                        pass
                    continue

                try:
                    gdict[k] = float(v)
                except ValueError:
                    pass

        geo = cls(_skip_autoload=True, **gdict)
        geo.set_assignment(slots_per_ml, ml0, ml1, apply_map=apply_map)
        return geo

    @classmethod
    def enforce_exclusive_active_tdcs(
        cls,
        geos: List["Geometry"],
        *,
        keep_ncol: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Make active TDC IDs exclusive across a list of Geometry objects.

        Rule:
          - Sort by chamber_id ascending
          - Lowest chamber_id "owns" a TDC
          - If a later (higher chamber_id) geo has the same active TDC (tdc_id>=0 and ncol>0),
            remove it from the later geo by setting tdc_id=-1.

        keep_ncol:
          - True: (-1, ncol)  keeps packing unchanged (recommended)
          - False: (-1, 0)    frees columns (packing changes)

        This mutates the Geometry objects in-place and re-applies set_assignment(apply_map=True).
        """
        if not geos:
            return

        # deterministic: lowest chamber_id wins
        geos_sorted = sorted(geos, key=lambda g: int(getattr(g, "chamber_id", 0)))

        seen: set[int] = set()

        for g in geos_sorted:
            # Ensure internal maps match current assignment before checking
            g.set_assignment(g.slots_per_ml, g.ml0, g.ml1, apply_map=True)

            removed: List[Tuple[str, int]] = []

            def _scrub(slot_list: List[Tuple[int, int]], ml_name: str) -> List[Tuple[int, int]]:
                out: List[Tuple[int, int]] = []
                for (tdc_id, ncol) in slot_list:
                    t = int(tdc_id)
                    n = max(0, int(ncol))

                    if t >= 0 and n > 0 and t in seen:
                        out.append((-1, n if keep_ncol else 0))
                        removed.append((ml_name, t))
                    else:
                        out.append((t, n))
                        if t >= 0 and n > 0:
                            seen.add(t)
                return out

            ml0_new = _scrub(g.ml0, "ML0")
            ml1_new = _scrub(g.ml1, "ML1")

            if ml0_new != g.ml0 or ml1_new != g.ml1:
                g.set_assignment(g.slots_per_ml, ml0_new, ml1_new, apply_map=True)

            if verbose and removed:
                print(f"[Geometry] chamber_id={g.chamber_id}: removed duplicated TDCs {removed}")

