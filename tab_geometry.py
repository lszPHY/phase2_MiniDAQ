# tab_geometry.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Mapping

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

from geometry import Geometry
from geometry_dialog import GeometryConfigDialog
from rt_function import RTFunction
from TrackFit import TrackFit


# =============================================================================
# Drawing item
# =============================================================================

class _TubeItem(QtWidgets.QGraphicsEllipseItem):
    __slots__ = ("layer", "col", "tdc_slot_idx")

    def __init__(self, x_mm: float, y_mm: float, r_mm: float):
        super().__init__(x_mm - r_mm, y_mm - r_mm, 2 * r_mm, 2 * r_mm)
        self.setPen(pg.mkPen(color=(80, 80, 80), width=1.0))
        self.setBrush(pg.mkBrush(255, 255, 255))
        self.layer = -1
        self.col = -1
        self.tdc_slot_idx = -1


# =============================================================================
# Main tab
# =============================================================================

class tab_geometry(QtCore.QObject):
    def __init__(
        self,
        parent_widget,
        geo: Geometry,
        backend,
        chamber_id: int,
        n_tdcs: int = 40,
        default_filename: str = "geometry.txt",
        rt: Optional[RTFunction] = None,
        rt_tmin: float = 0.0,
        rt_tmax: float = 200.0,
    ):
        super().__init__(parent_widget)
        self.parent = parent_widget
        self.backend = backend
        self.geo = geo
        self.chamber_id = int(chamber_id)
        self.default_filename = str(default_filename)

        # ---------------- RT + drift-circle overlay ----------------
        self._rt: Optional[RTFunction] = rt

        # ---- cached pens/brushes (avoid recreate per event) ----
        self._pen_blue = pg.mkPen(color=(0, 0, 255), width=1.0)
        self._brush_blue = pg.mkBrush(0, 0, 255, 80)
        self._pen_red_radius = pg.mkPen(color=(220, 0, 0), width=1.2)
        self._brush_red_radius = pg.mkBrush(255, 0, 0, 80)
        self._brush_red = pg.mkBrush(255, 0, 0)
        self._brush_green = pg.mkBrush(0, 255, 0)
        self._pen_track = pg.mkPen(color=(0, 0, 180), width=2.4)

        # drift objects
        self._drift_circle_items: List[QtWidgets.QGraphicsEllipseItem] = []
        self._drift_text_items: List[pg.TextItem] = []
        self._removed_text_items: List[pg.TextItem] = []
        self._removed_tubes: set[Tuple[int, int]] = set()
        self._track_line_items: List[QtWidgets.QGraphicsLineItem] = []

        # Track fitter for automatic valid-event rendering.
        self._trackfit: Optional[TrackFit] = None
        if self._rt is not None and TrackFit.available():
            try:
                self._trackfit = TrackFit.from_rt(
                    self._rt,
                    n_lut=2048,
                    min_hits=6,
                    fit_only_kept_hits=True,
                )
            except Exception as e:
                print(f"[TrackFit] Failed to initialize chamber {self.chamber_id}: {e}")

        # Use Geometry's loaded assignment if available; otherwise fall back to defaults.
        self.slots_per_ml = int(getattr(self.geo, "slots_per_ml", 10))
        if getattr(self.geo, "ml0", None) and getattr(self.geo, "ml1", None):
            self.ml0_slots = list(self.geo.ml0[: self.slots_per_ml])
            self.ml1_slots = list(self.geo.ml1[: self.slots_per_ml])
        else:
            self.ml0_slots = [(i, 6) for i in range(self.slots_per_ml)]
            self.ml1_slots = [(10 + i, 6) for i in range(self.slots_per_ml)]

        print(
            f"[tab_geometry init] chamber={getattr(self.geo,'chamber_id',-1)} "
            f"slots_per_ml={self.slots_per_ml} ml0_slots={self.ml0_slots} ml1_slots={self.ml1_slots}"
        )

        # lookup + highlight
        self._tube_items: List[_TubeItem] = []
        self._tube_by_lc: Dict[Tuple[int, int], _TubeItem] = {}
        self._highlighted: set[Tuple[int, int]] = set()

        # label caches
        self._tdc_text_items: List[pg.TextItem] = []
        self._ch_text_items: List[pg.TextItem] = []
        self._show_channel_ids = False

        # packed starts per ML (computed in _sync_geo_tdc_map)
        self._ml_slot_starts: Dict[int, List[int]] = {0: [], 1: []}
        self._ml_slot_coverage: Dict[int, Dict[int, int]] = {0: {}, 1: {}}
        self._ml_slots_expanded: Dict[int, List[Tuple[int, int, int, int]]] = {0: [], 1: []}

        self._startup_load_error = ""

        self._sync_geo_tdc_map()
        self._build_ui()
        self._redraw()

        # global nav (backend-owned)
        if self.backend is not None and hasattr(self.backend, "event_changed"):
            self.backend.event_changed.connect(self._on_global_event_changed)

        if self.backend is not None and hasattr(self.backend, "analysis_1hz"):
            self.backend.analysis_1hz.connect(self._on_decode_tick)

        # initial sync to current backend event (if any)
        if self.backend is not None and hasattr(self.backend, "current_event"):
            # Backend payload (new):
            #   (Event_raw, (time_by_chamber, valid_by_chamber)) or (None, None)
            payload0 = self.backend.current_event()
            self._on_global_event_changed(payload0)
            ev0, _, _ = self._unpack_payload(payload0)
            self._update_event_nav_ui_global(ev0, payload0=payload0)
        else:
            self._update_event_nav_ui_global(None, payload0=None)

    # =============================================================================
    # Payload unpack helper
    # =============================================================================

    @staticmethod
    def _unpack_payload(payload: Any) -> Tuple[Optional[Any], Optional[Dict[int, Any]], Optional[Dict[int, bool]]]:
        """
        Payload format from backend:

            (Event_raw, (time_by_chamber, valid_by_chamber))
            or
            (None, None)

        where:
            time_by_chamber: {cid: TimeWindowResult}  (t_corr_ns, keep_mask, ...)
            valid_by_chamber: {cid: bool}
        """
        if payload is None:
            return None, None, None

        if isinstance(payload, tuple) and len(payload) == 2:
            ev = payload[0]
            q = payload[1]

            if q is None:
                return ev, None, None

            if isinstance(q, tuple) and len(q) == 2:
                time_by_ch, valid_by_ch = q
                return ev, time_by_ch, valid_by_ch

            # legacy fallback: (Event_raw, dict)
            if isinstance(q, dict):
                # treat as time_by_ch only
                return ev, q, None

        # legacy fallback: Event only
        return payload, None, None

    # =============================================================================
    # Debug
    # =============================================================================

    def _debug_print_active_tdcs(self, tag: str = ""):
        active = [i for i, a in enumerate(self.geo.isActiveTDC) if int(a) == 1]
        ml0 = [t for t in active if int(self.geo.TDC_ML[t]) == 0]
        ml1 = [t for t in active if int(self.geo.TDC_ML[t]) == 1]

        prefix = f"[tab_geometry]{' ' + tag if tag else ''}"
        print(f"{prefix} ActiveTDC count={len(active)} active={active}")
        print(f"{prefix}   ML0 active={ml0}")
        print(f"{prefix}   ML1 active={ml1}")
        for t in active[:10]:
            print(f"{prefix}   TDC {t:02d}: ML={int(self.geo.TDC_ML[t])} COLSTART={int(self.geo.TDC_COL[t])}")

    # =============================================================================
    # Packed-slot helpers
    # =============================================================================

    @staticmethod
    def _slot_starts(slots: List[Tuple[int, int]]) -> List[int]:
        starts: List[int] = []
        acc = 0
        for _, ncol in slots:
            starts.append(acc)
            acc += max(0, int(ncol))
        return starts

    def _slot_start(self, ml_id: int, slot_idx: int) -> int:
        starts = self._ml_slot_starts.get(int(ml_id), [])
        if 0 <= int(slot_idx) < len(starts):
            return int(starts[int(slot_idx)])
        return 0

    # =============================================================================
    # Mapping
    # =============================================================================

    def _sync_geo_tdc_map(self):
        self._ml_slot_starts = {
            0: self._slot_starts(self.ml0_slots),
            1: self._slot_starts(self.ml1_slots),
        }

        active: List[int] = []
        ml: List[int] = []
        colstart: List[int] = []

        for ml_id, slots in [(0, self.ml0_slots), (1, self.ml1_slots)]:
            starts = self._ml_slot_starts[ml_id]
            for slot_idx, (tdc_id, ncol) in enumerate(slots):
                tdc_id = int(tdc_id)
                ncol = max(0, int(ncol))
                if tdc_id >= 0 and ncol > 0:
                    active.append(tdc_id)
                    ml.append(int(ml_id))
                    colstart.append(int(starts[slot_idx]))

        self.geo.configure_tdc_map(active, ml, colstart, strict_duplicates=False)

        self._ml_slot_coverage = {0: {}, 1: {}}
        self._ml_slots_expanded = {0: [], 1: []}

        for ml_id, slots in [(0, self.ml0_slots), (1, self.ml1_slots)]:
            starts = self._ml_slot_starts[ml_id]
            for slot_idx, (tdc_id, ncol) in enumerate(slots):
                tdc_id = int(tdc_id)
                ncol = max(0, int(ncol))
                cs = int(starts[slot_idx])

                self._ml_slots_expanded[ml_id].append((slot_idx, tdc_id, cs, ncol))

                if tdc_id < 0 or ncol <= 0:
                    continue

                for c in range(cs, cs + ncol):
                    if 0 <= c < int(self.geo.MAX_TUBE_COLUMN):
                        self._ml_slot_coverage[ml_id].setdefault(c, slot_idx)

        self._debug_print_active_tdcs("sync_geo_tdc_map")

    # =============================================================================
    # UI
    # =============================================================================

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self.parent)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        self.btn_config = QtWidgets.QPushButton("Geometry + Slots...")
        self.btn_config.clicked.connect(self._open_geometry_config_dialog)
        top.addWidget(self.btn_config)

        self.btn_prev_ev = QtWidgets.QPushButton("Prev event")
        self.btn_next_ev = QtWidgets.QPushButton("Next event")
        top.addWidget(self.btn_prev_ev)
        top.addWidget(self.btn_next_ev)
        self.btn_prev_ev.clicked.connect(self._on_prev_event)
        self.btn_next_ev.clicked.connect(self._on_next_event)

        self.cb_show_ch = QtWidgets.QCheckBox("Show channel ID in tubes")
        self.cb_show_ch.setChecked(False)
        self.cb_show_ch.stateChanged.connect(self._on_toggle_channel_ids)
        top.addWidget(self.cb_show_ch)

        self.lab_status = QtWidgets.QLabel("")
        top.addWidget(self.lab_status, 1)

        self.lab_event = QtWidgets.QLabel("")
        top.addWidget(self.lab_event, 1)

        self.view = pg.PlotWidget()
        self.view.showGrid(x=True, y=True, alpha=0.2)
        self.view.setLabel("bottom", "x", units="mm")
        self.view.setLabel("left", "y", units="mm")
        self.view.setAspectLocked(True, ratio=1.0)
        self.view.setMenuEnabled(False)
        layout.addWidget(self.view, 1)

    def _on_toggle_channel_ids(self, state: int):
        self._show_channel_ids = bool(state)
        self._update_channel_labels()

    def _open_geometry_config_dialog(self):
        dlg = GeometryConfigDialog(
            self.parent,
            geo=self.geo,
            slots_per_ml=self.slots_per_ml,
            ml0_slots=self.ml0_slots,
            ml1_slots=self.ml1_slots,
            default_path=self.default_filename,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        new_geo = dlg.result_geometry()
        self.geo._copy_from(new_geo)          # keep identity

        self.slots_per_ml = dlg.result_slots_per_ml()
        self.ml0_slots    = dlg.result_ml0_slots()
        self.ml1_slots    = dlg.result_ml1_slots()

        # IMPORTANT: apply the dialog slots into the shared geo object
        self.geo.set_assignment(self.slots_per_ml, self.ml0_slots, self.ml1_slots, apply_map=True)

        self._sync_geo_tdc_map()

        if self.backend is not None and hasattr(self.backend, "clear_event_cache"):
            self.backend.clear_event_cache()
        if self.backend is not None and hasattr(self.backend, "reset_fit_spectra"):
            self.backend.reset_fit_spectra()
        else:
            self.clear_hit_highlight()

        self._redraw()

    # =============================================================================
    # Drawing helpers
    # =============================================================================

    def _tube_brush_for_slot(self, slot_idx: int):
        if slot_idx < 0:
            return pg.mkBrush(255, 255, 255)
        return pg.mkBrush(255, 255, 255) if (slot_idx % 2 == 0) else pg.mkBrush(220, 220, 220)

    def _slot_for_tube(self, layer: int, col: int) -> int:
        ml = self.geo.multilayer_from_layer(layer)
        return self._ml_slot_coverage.get(ml, {}).get(col, -1)

    def _clear_scene(self):
        for it in self._tube_items:
            self.view.removeItem(it)
        self._tube_items.clear()

        for t in self._tdc_text_items:
            self.view.removeItem(t)
        self._tdc_text_items.clear()

        for t in self._ch_text_items:
            self.view.removeItem(t)
        self._ch_text_items.clear()

        self._clear_drift_circles()

    def _add_tdc_labels(self):
        y0_top = self.geo.get_hit_xy(3, 0)[1]
        y1_top = self.geo.get_hit_xy(7, 0)[1]
        y0 = y0_top + 2.2 * self.geo.radius
        y1 = y1_top + 2.2 * self.geo.radius

        for ml_id, ylab in [(0, y0), (1, y1)]:
            for slot_idx, tdc_id, col_start, ncol in self._ml_slots_expanded[ml_id]:
                if int(tdc_id) < 0 or int(ncol) <= 0:
                    continue
                cols = [
                    c for c in range(int(col_start), int(col_start) + int(ncol))
                    if 0 <= c < int(self.geo.MAX_TUBE_COLUMN)
                ]
                if not cols:
                    continue
                rep_layer = ml_id * int(self.geo.MAX_TDC_LAYER)
                xs: List[float] = []
                for c in cols:
                    x, _ = self.geo.get_hit_xy(rep_layer, c)
                    if x >= 0:
                        xs.append(x)
                if not xs:
                    continue
                xlab = float(sum(xs) / len(xs))
                t = pg.TextItem(text=f"TDC {tdc_id:02d}", color=(0, 0, 0), anchor=(0.5, 0.0))
                t.setPos(xlab, ylab)
                self.view.addItem(t)
                self._tdc_text_items.append(t)

    # =============================================================================
    # Per-chamber hits (NEW, NO fallback by design)
    # =============================================================================

    def _hits_for_this_chamber(self, ev: Any) -> Tuple[Any, ...]:
        """
        Returns this tab's ChamberBlock.hits (tuple of Hit) or ().
        Event model:
          ev.chambers: Dict[int, ChamberBlock]
          ev.chambers[cid].hits: Tuple[Hit, ...]
        """
        if ev is None:
            return ()
        chambers = getattr(ev, "chambers", None)
        if not isinstance(chambers, dict) or not chambers:
            return ()
        block = chambers.get(int(self.chamber_id))
        if block is None:
            return ()
        hits = getattr(block, "hits", None)
        return hits if hits else ()

    def _timewindow_for_this_chamber(self, time_by_ch: Any) -> Optional[Any]:
        """
        time_by_ch is expected to be dict {cid: TimeWindowResult}.
        """
        if time_by_ch is None or not isinstance(time_by_ch, dict):
            return None
        return time_by_ch.get(int(self.chamber_id))

    def _valid_for_this_chamber(self, valid_by_ch: Any) -> Optional[bool]:
        """
        valid_by_ch is expected to be dict {cid: bool}.
        """
        if valid_by_ch is None or not isinstance(valid_by_ch, dict):
            return None
        return bool(valid_by_ch.get(int(self.chamber_id), True))

    # =============================================================================
    # Labels / redraw
    # =============================================================================

    def _update_channel_labels(self):
        for t in self._ch_text_items:
            self.view.removeItem(t)
        self._ch_text_items.clear()

        if not self._show_channel_ids:
            return

        view = self.view
        geo = self.geo
        addItem = view.addItem

        for item in self._tube_items:
            layer = item.layer
            col = item.col
            ml = geo.multilayer_from_layer(layer)
            slot_idx = self._slot_for_tube(layer, col)
            if slot_idx < 0:
                continue

            tdc_id, ncol = (self.ml0_slots[slot_idx] if ml == 0 else self.ml1_slots[slot_idx])
            tdc_id = int(tdc_id)
            ncol = max(0, int(ncol))
            if tdc_id < 0 or ncol <= 0:
                continue

            col_start = self._slot_start(ml, slot_idx)
            local_col = int(col) - int(col_start)
            local_layer = int(layer) - int(ml) * int(geo.MAX_TDC_LAYER)

            if not (0 <= local_col < ncol and 0 <= local_layer <= int(geo.MAX_TDC_LAYER) - 1):
                continue

            ch = geo.channel_id_from_local(local_layer, local_col)
            if ch < 0:
                continue

            x, y = geo.get_hit_xy(layer, col)
            txt = pg.TextItem(text=f"{ch}", color=(0, 0, 0), anchor=(0.5, 0.5))
            txt.setPos(x, y)
            addItem(txt)
            self._ch_text_items.append(txt)

    def _redraw(self):
        self._clear_scene()
        self._tube_by_lc.clear()
        self._highlighted.clear()

        geo = self.geo
        view = self.view
        addItem = view.addItem

        n_draw = 0
        xs: List[float] = []
        ys: List[float] = []

        for layer in range(int(geo.MAX_TUBE_LAYER)):
            for col in range(int(geo.MAX_TUBE_COLUMN)):
                x, y = geo.get_hit_xy(layer, col)
                if x < 0:
                    continue

                slot_idx = self._slot_for_tube(layer, col)
                if slot_idx < 0:
                    continue

                it = _TubeItem(x, y, float(geo.radius))
                it.layer = layer
                it.col = col
                it.tdc_slot_idx = slot_idx
                it.setBrush(self._tube_brush_for_slot(slot_idx))

                addItem(it)
                self._tube_items.append(it)
                self._tube_by_lc[(layer, col)] = it

                n_draw += 1
                xs.append(x)
                ys.append(y)

        if xs and ys:
            xmin, xmax = min(xs) - 2 * geo.radius, max(xs) + 2 * geo.radius
            ymin, ymax = min(ys) - 2 * geo.radius, max(ys) + 4 * geo.radius
            view.setXRange(xmin, xmax, padding=0.0)
            view.setYRange(ymin, ymax, padding=0.0)

        self._add_tdc_labels()
        self._update_channel_labels()

        tot0 = sum(max(0, int(n)) for _, n in self.ml0_slots[: self.slots_per_ml])
        tot1 = sum(max(0, int(n)) for _, n in self.ml1_slots[: self.slots_per_ml])
        self.lab_status.setText(
            f"Draw tubes: {n_draw} | slots_per_ml={self.slots_per_ml} | "
            f"MAX_TDC={geo.MAX_TDC} MAX_TUBE_COLUMN={geo.MAX_TUBE_COLUMN} | "
            f"ML0 sum(ncol)={tot0} ML1 sum(ncol)={tot1}"
        )

    # =============================================================================
    # Drift circles (optimized)
    # =============================================================================

    def _clear_drift_circles(self):
        view = self.view

        for it in self._drift_circle_items:
            view.removeItem(it)
        self._drift_circle_items.clear()

        for t in self._drift_text_items:
            view.removeItem(t)
        self._drift_text_items.clear()

        for t in self._removed_text_items:
            view.removeItem(t)
        self._removed_text_items.clear()

        for it in self._track_line_items:
            view.removeItem(it)
        self._track_line_items.clear()

        # restore removed tube brushes back to normal
        tube_by_lc = self._tube_by_lc
        for (L, C) in list(self._removed_tubes):
            it = tube_by_lc.get((L, C))
            if it is not None:
                it.setBrush(self._tube_brush_for_slot(it.tdc_slot_idx))
        self._removed_tubes.clear()

    @staticmethod
    def _line_segment_in_box(
        nx: float,
        ny: float,
        c: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        eps = 1e-12
        pts: List[Tuple[float, float]] = []

        if abs(ny) > eps:
            for x in (xmin, xmax):
                y = (c - nx * x) / ny
                if ymin - 1e-9 <= y <= ymax + 1e-9:
                    pts.append((float(x), float(y)))

        if abs(nx) > eps:
            for y in (ymin, ymax):
                x = (c - ny * y) / nx
                if xmin - 1e-9 <= x <= xmax + 1e-9:
                    pts.append((float(x), float(y)))

        # deduplicate near-identical intersections
        uniq: List[Tuple[float, float]] = []
        for p in pts:
            if all((abs(p[0] - q[0]) > 1e-9 or abs(p[1] - q[1]) > 1e-9) for q in uniq):
                uniq.append(p)

        if len(uniq) < 2:
            return None
        if len(uniq) == 2:
            return uniq[0], uniq[1]

        # more than 2 intersections can happen on corners; pick farthest pair
        best_i = 0
        best_j = 1
        best_d2 = -1.0
        for i in range(len(uniq)):
            xi, yi = uniq[i]
            for j in range(i + 1, len(uniq)):
                xj, yj = uniq[j]
                d2 = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
                if d2 > best_d2:
                    best_d2 = d2
                    best_i = i
                    best_j = j
        return uniq[best_i], uniq[best_j]

    def _fit_result_from_timewindow(self, ev: Any, time_by_ch: Any):
        if self._trackfit is None or ev is None or not isinstance(time_by_ch, dict):
            return None

        hits = self._hits_for_this_chamber(ev)
        tw = self._timewindow_for_this_chamber(time_by_ch)
        if not hits or tw is None:
            return None

        t_corr = getattr(tw, "t_corr_ns", None)
        keep = getattr(tw, "keep_mask", None)
        if t_corr is None:
            return None

        fit = self._trackfit.fit_chamber_hits(
            chamber_id=self.chamber_id,
            hits=hits,
            t_corr_ns=t_corr,
            keep_mask=keep,
        )
        if not fit.ok:
            return None

        vals = fit.values
        if not vals:
            return None

        return fit

    def set_rt(self, rt: Optional[RTFunction]) -> None:
        """
        Update the RT model used by this geometry tab and refresh the current view.
        """
        self._rt = rt
        self._trackfit = None
        if self._rt is not None and TrackFit.available():
            try:
                self._trackfit = TrackFit.from_rt(
                    self._rt,
                    n_lut=2048,
                    min_hits=6,
                    fit_only_kept_hits=True,
                )
            except Exception as e:
                print(f"[TrackFit] Failed to initialize chamber {self.chamber_id}: {e}")

        if self.backend is not None and hasattr(self.backend, "current_event"):
            try:
                payload = self.backend.current_event()
            except Exception:
                payload = None
            self._on_global_event_changed(payload)

    def _draw_track_line_from_fit_values(self, vals: Mapping[str, float]):
        if not vals:
            return

        try:
            nx = float(vals.get("nx"))
            ny = float(vals.get("ny"))
            c = float(vals.get("c"))
        except Exception:
            return

        (x0, x1), (y0, y1) = self.view.getViewBox().viewRange()
        seg = self._line_segment_in_box(nx, ny, c, float(x0), float(x1), float(y0), float(y1))
        if seg is None:
            return

        (xa, ya), (xb, yb) = seg
        line = QtWidgets.QGraphicsLineItem(xa, ya, xb, yb)
        line.setPen(self._pen_track)
        line.setZValue(30)
        self.view.addItem(line)
        self._track_line_items.append(line)

    def _draw_drift_circles_from_timewindow(
        self,
        ev: Any,
        time_by_ch: Any,
        *,
        t0_ns_shift: float = 0.0,
        fit_used_mask: Optional[np.ndarray] = None,
    ):
        """
        Draw ONLY this tab's chamber hits.
        NO fallback by design:
          - Event provides ev.chambers[self.chamber_id].hits
          - Backend provides time_by_ch[self.chamber_id] with keep_mask/t_corr_ns aligned to those hits
          - Hit provides x/y/layer/col
        """
        if self._rt is None or ev is None or time_by_ch is None:
            return

        hits = self._hits_for_this_chamber(ev)
        tw = self._timewindow_for_this_chamber(time_by_ch)
        if not hits or tw is None:
            return

        keep = getattr(tw, "keep_mask", None)
        t_corr = getattr(tw, "t_corr_ns", None)
        if keep is None or t_corr is None:
            return

        # Expect numpy arrays from backend; only convert if not.
        if not isinstance(keep, np.ndarray):
            keep = np.asarray(keep, dtype=bool)
        if not isinstance(t_corr, np.ndarray):
            t_corr = np.asarray(t_corr, dtype=np.float64)

        if keep.shape[0] != len(hits) or t_corr.shape[0] != len(hits):
            return

        if fit_used_mask is not None:
            if not isinstance(fit_used_mask, np.ndarray):
                fit_used_mask = np.asarray(fit_used_mask, dtype=bool)
            if fit_used_mask.shape[0] != len(hits):
                fit_used_mask = None

        # Compute real drift time for these hits only:
        #   t_real = t_corr - t0_fit
        t_real = t_corr - float(t0_ns_shift)
        t_rt_min = float(self._rt.T0)
        t_rt_max = float(self._rt.Tmax)
        in_rt = np.isfinite(t_real) & (t_real >= t_rt_min) & (t_real <= t_rt_max)
        if not np.all(in_rt):
            # With strict adaptive t0 bounds this should not happen for fitted valid events.
            return

        r_sel = self._rt.r_of_t(t_real)
        r_sel = np.clip(r_sel, 0.0, float(self.geo.max_drift_dist))

        view = self.view
        addItem = view.addItem
        tube_by_lc = self._tube_by_lc

        pen_blue = self._pen_blue
        brush_blue = self._brush_blue
        pen_red_radius = self._pen_red_radius
        brush_red_radius = self._brush_red_radius
        brush_red = self._brush_red

        for i, (h, is_kept, tns_raw, tns_real, rr) in enumerate(zip(hits, keep, t_corr, t_real, r_sel)):
            x = float(getattr(h, "x", -1.0))
            y = float(getattr(h, "y", -1.0))
            layer = int(getattr(h, "layer", -1))
            col = int(getattr(h, "col", -1))
            if x < 0 or y < 0 or layer < 0 or col < 0:
                continue

            tube = tube_by_lc.get((layer, col))

            if bool(is_kept):
                rr = float(rr)
                circ = QtWidgets.QGraphicsEllipseItem(x - rr, y - rr, 2 * rr, 2 * rr)
                fit_kept = True
                if fit_used_mask is not None:
                    fit_kept = bool(fit_used_mask[i])

                if fit_kept:
                    circ.setPen(pen_blue)
                    circ.setBrush(brush_blue)
                else:
                    # Excluded by fitter hit mask -> show red radius.
                    circ.setPen(pen_red_radius)
                    circ.setBrush(brush_red_radius)
                circ.setZValue(10)
                addItem(circ)
                self._drift_circle_items.append(circ)

                lab = pg.TextItem(text=f"{float(tns_real):.1f}", color=(0, 0, 0), anchor=(0.5, 0.5))
                lab.setPos(x, y)
                lab.setZValue(20)
                addItem(lab)
                self._drift_text_items.append(lab)
            else:
                if tube is not None:
                    tube.setBrush(brush_red)
                    self._removed_tubes.add((layer, col))

                lab = pg.TextItem(text=f"{float(tns_real):.1f}", color=(0, 0, 0), anchor=(0.5, 0.5))
                lab.setPos(x, y)
                lab.setZValue(20)
                addItem(lab)
                self._removed_text_items.append(lab)

    # =============================================================================
    # Hit highlight + global event browsing
    # =============================================================================

    def clear_hit_highlight(self):
        tube_by_lc = self._tube_by_lc
        for (L, C) in list(self._highlighted):
            it = tube_by_lc.get((L, C))
            if it is not None:
                it.setBrush(self._tube_brush_for_slot(it.tdc_slot_idx))
        self._highlighted.clear()

    def highlight_event_green(self, ev: Any):
        if ev is None:
            return
        self.clear_hit_highlight()

        hits = self._hits_for_this_chamber(ev)
        if not hits:
            return

        tube_by_lc = self._tube_by_lc
        green = self._brush_green
        highlighted = self._highlighted

        for h in hits:
            L = int(getattr(h, "layer", -1))
            C = int(getattr(h, "col", -1))
            if L < 0 or C < 0:
                continue
            it = tube_by_lc.get((L, C))
            if it is None:
                continue
            it.setBrush(green)
            highlighted.add((L, C))

    def _update_event_nav_ui_global(self, ev: Any = None, *, payload0: Any = None, payload: Any = None):
        """
        UI string shows global navigation state, but hit counts are tab-local.
        If payload is provided, we can also show this chamber's validity.
        """
        if self.backend is None:
            self.btn_prev_ev.setEnabled(False)
            self.btn_next_ev.setEnabled(False)
            self.lab_event.setText("No backend")
            return

        idx = self.backend.current_index() if hasattr(self.backend, "current_index") else -1
        n = self.backend.cache_size() if hasattr(self.backend, "cache_size") else 0
        buffered = self.backend.events_buffered() if hasattr(self.backend, "events_buffered") else 0

        has_prev = idx > 0
        has_next_cached = (0 <= idx < n - 1)
        has_next = has_next_cached or (buffered > 0)

        self.btn_prev_ev.setEnabled(bool(has_prev))
        self.btn_next_ev.setEnabled(bool(has_next))

        if ev is None and hasattr(self.backend, "current_event"):
            payload_now = self.backend.current_event()
            ev, time_by_ch, valid_by_ch = self._unpack_payload(payload_now)
        else:
            # if caller passed a payload, parse it; else validity unknown
            payload_now = payload or payload0
            _, time_by_ch, valid_by_ch = self._unpack_payload(payload_now) if payload_now is not None else (None, None, None)

        if ev is None or idx < 0 or n <= 0:
            self.lab_event.setText(f"No event selected | buffered={buffered}")
            return

        n_tab = len(self._hits_for_this_chamber(ev))

        # chamber validity (optional)
        chamber_valid = self._valid_for_this_chamber(valid_by_ch)
        valid_txt = ""
        mode = "auto"
        if chamber_valid is not None:
            valid_txt = f" | chamber_valid={bool(chamber_valid)}"
            mode = "auto: drift+track" if bool(chamber_valid) else "auto: highlight"

        self.lab_event.setText(
            f"Event {idx + 1}/{n} | "
            f"eid20={getattr(ev, 'event_id20', -1)} | "
            f"hits(this tab)={n_tab} | buffered={buffered} | mode={mode}"
            f"{valid_txt}"
        )

    def _on_prev_event(self):
        if self.backend is not None and hasattr(self.backend, "prev_event"):
            self.backend.prev_event()

    def _on_next_event(self):
        if self.backend is not None and hasattr(self.backend, "next_event"):
            self.backend.next_event()

    # optional convenience
    def map_hit_to_wire(self, tdc_id: int, ch_id: int) -> Tuple[float, float, int, int]:
        x, y, layer, col = self.geo.wire_center_from_hit(int(tdc_id), int(ch_id))
        return x, y, layer, col

    @QtCore.pyqtSlot(object)
    def _on_global_event_changed(self, payload):
        """
        payload: (Event_raw, (time_by_chamber, valid_by_chamber)) or (None,None)

        This tab draws ONLY hits belonging to self.chamber_id via ev.chambers[cid].hits.
        NO fallback by design.
        """
        ev, time_by_ch, valid_by_ch = self._unpack_payload(payload)

        # clear overlays first
        self._clear_drift_circles()

        if ev is None:
            self.clear_hit_highlight()
            self._update_event_nav_ui_global(None, payload=payload)
            return

        # chamber validity
        ch_valid = True
        if isinstance(valid_by_ch, dict):
            ch_valid = valid_by_ch.get(self.chamber_id, True)

        # if invalid ? only highlight hits
        if not ch_valid:
            self.highlight_event_green(ev)
            self._update_event_nav_ui_global(ev, payload=payload)
            return

        # valid events: always draw drift radii + track (auto mode)
        if self._rt is not None and isinstance(time_by_ch, dict):
            self.clear_hit_highlight()
            fit = self._fit_result_from_timewindow(ev, time_by_ch)
            if fit is None:
                # No physically consistent fit/t0 for this event -> fallback highlight.
                self.highlight_event_green(ev)
                self._update_event_nav_ui_global(ev, payload=payload)
                return

            fit_vals = fit.values
            t0_shift = float(fit_vals.get("t0_ns", 0.0))
            fit_used_mask = None
            raw_mask = fit_vals.get("used_mask_input", None)
            if raw_mask is not None:
                try:
                    fit_used_mask = np.asarray(raw_mask, dtype=bool)
                except Exception:
                    fit_used_mask = None

            self._draw_drift_circles_from_timewindow(
                ev,
                time_by_ch,
                t0_ns_shift=t0_shift,
                fit_used_mask=fit_used_mask,
            )
            self._draw_track_line_from_fit_values(fit_vals)
        else:
            self.highlight_event_green(ev)

        self._update_event_nav_ui_global(ev, payload=payload)

    @QtCore.pyqtSlot(object)
    def _on_decode_tick(self, snap):
        # 1Hz UI refresh for buffered count
        self._update_event_nav_ui_global(None)
