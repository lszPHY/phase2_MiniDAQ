# backend.py
# Integrates CaptureThread (.dat writing + byte queue) and DecodeThread (event build + stats)
#
# UPDATED behavior (Option C, chamber-based):
#   - event_changed emits (Event_raw, (time_by_chamber, valid_by_chamber)) or (None, None)
#   - cache stores RAW Events (not filtered)
#   - per-chamber checks are computed on-demand PER EVENT (cached per event_id20)
#
# Notes:
#   - Event has NO global hits list. It contains:
#       ev.chambers: Dict[int, ChamberBlock]
#       ev.chambers[cid].hits: Tuple[Hit, ...]
#
# Why this structure (original explanation, still true):
#   - DecodeThread stays fast and simple: it only builds Events and basic histograms.
#   - Backend owns "analysis" logic that can change often (time slew, filtering, QA).
#   - GUI tabs all receive the SAME Event object (one broadcast), but each tab only
#     reads its own chamber_id entry from the emitted dictionaries (no mixing).
#   - Caching is done by event_id20 so back/forward browsing is instant.

from __future__ import annotations

import os
import time
import queue
from types import SimpleNamespace
from typing import Optional, Any, Dict, List, Tuple

from PyQt5 import QtCore
import numpy as np

from CaptureThread import CaptureThread
from ReplayThread import DatReplayThread
from DecodeThread import DecodeThread, EventBuffer
from GeoRouter import GeoRouter

# NEW: class-only API (no module-level functions)
from EventQuality import TimeSlewASD2, EventQualityChecker
from TrackFit import TrackFit, TrackFitResult


def _timestamp_yyyymmdd_hhmmss() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


TimeByChamber = Dict[int, EventQualityChecker.TimeWindowResult]
ValidByChamber = Dict[int, bool]
FitByChamber = Dict[int, TrackFitResult]

# Payload emitted by event_changed:
#   (Event_raw, (time_by_chamber, valid_by_chamber)) or (None, None)
EventPayload = Tuple[Optional[Any], Optional[Tuple[TimeByChamber, ValidByChamber]]]


class Backend(QtCore.QObject):
    """
    Backend wires:
      CaptureThread -> analysis_q -> DecodeThread

    Exposes:
      - message(str): capture messages
      - stats(total_packets, lost_total, total_bytes, filename): capture stats
      - analysis_1hz(DecodeSnapshot): decode stats + histograms (valid events only)

    Provides:
      - pop_event() to retrieve valid events from EventBuffer

    Geometry registry (GUI-side, NOT used by DecodeThread):
      - set_geometries_from_list([geo0, geo1, ...])
      - get_geometry(chamber_id)
      - geometry_count()

    Global event navigation:
      - event_changed((Event_raw, (time_by_chamber, valid_by_chamber)) | (None,None))
      - next_event(), prev_event(), goto_event(idx)
      - current_event(), current_index(), cache_size()
      - clear_event_cache()

    Original intent (still the same):
      - CaptureThread does I/O and pushes raw bytes into a bounded queue.
      - DecodeThread does low-level decoding and Event building (fast, no heavy analysis).
      - Backend does higher-level analysis/QA on-demand, and emits payloads to GUI tabs.
    """

    message = QtCore.pyqtSignal(str)
    stats = QtCore.pyqtSignal(int, int, int, str)
    capture_started = QtCore.pyqtSignal()
    run_finished = QtCore.pyqtSignal()
    analysis_1hz = QtCore.pyqtSignal(object)  # DecodeSnapshot

    # Emits: (Event_raw, (time_by_chamber, valid_by_chamber)) or (None, None)
    event_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        # bounded queue so decode can't eat RAM
        self._analysis_q: "queue.Queue[bytes]" = queue.Queue(maxsize=256)

        self._cap_thread: Optional[CaptureThread] = None
        self._replay_thread: Optional[DatReplayThread] = None
        self._dec_thread: Optional[DecodeThread] = None

        self._event_buf: Optional[EventBuffer] = None
        self._current_out_path: Optional[str] = None

        # chamber_id -> Geometry (GUI / mapping only)
        self._geos: Dict[int, Any] = {}

        # cached RAW Events (in the order user browsed)
        self._ev_cache: List[Any] = []
        self._ev_idx: int = -1

        # ---- quality checker (time slew + per-chamber validity) ----
        self._slew = TimeSlewASD2(coefficient=35.59, scale=0.0163)  # ASD2 always

        # You can tune thresholds here.
        # Note: even when checkTime=False, time slew correction is still computed
        # (t_corr_ns / keep_mask are always available for drift circles / debug).
        self._quality = EventQualityChecker(
            min_window_ns=10.0,
            max_window_ns=250.0,
            slew=self._slew,

            n_layers=8,
            min_total_hits=0,
            max_total_hits=16,
            min_layer_hit=6,
            max_hit_per_layer=2,
            max_unique_col_per_layer=2,

            checkTime=True,
            checkHitcount=True,
        )

        # Cache computed per-event quality and fit results keyed by event_id20.
        # Fit results are reused so chamber validity and spectra filling do not
        # rerun the same fit for the same chamber-event.
        self._qc_cache: Dict[int, Tuple[TimeByChamber, ValidByChamber]] = {}
        self._fit_cache: Dict[int, FitByChamber] = {}

        # ---- track-fit based drift-time spectra (t_real = t_corr_ns - t0_fit) ----
        self._trackfit: Optional[TrackFit] = None
        self._fit_tdc_bins = 512
        self._fit_tdc_tmin_ns = -100.0
        self._fit_tdc_tmax_ns = 300.0
        self._fit_tdc_binw_ns = (self._fit_tdc_tmax_ns - self._fit_tdc_tmin_ns) / float(self._fit_tdc_bins)
        self._fit_max_tdcs = 40
        self._fit_max_channels = 24
        self._fit_adc_bins = 256
        self._fit_adc_hist = np.zeros((self._fit_max_tdcs, self._fit_adc_bins), dtype=np.uint32)
        self._fit_adc_ch_hist = np.zeros(
            (self._fit_max_tdcs, self._fit_max_channels, self._fit_adc_bins),
            dtype=np.uint32,
        )
        self._fit_tdc_hist = np.zeros((self._fit_max_tdcs, self._fit_tdc_bins), dtype=np.uint32)
        self._fit_tdc_ch_hist = np.zeros(
            (self._fit_max_tdcs, self._fit_max_channels, self._fit_tdc_bins),
            dtype=np.uint32,
        )
        self._fit_resid_bins = 400
        self._fit_resid_min_mm = -5.0
        self._fit_resid_max_mm = 5.0
        self._fit_resid_binw_mm = (self._fit_resid_max_mm - self._fit_resid_min_mm) / float(self._fit_resid_bins)
        self._fit_resid_hist = np.zeros((self._fit_max_tdcs, self._fit_resid_bins), dtype=np.uint32)
        self._fit_resid_hist_by_chamber: Dict[int, np.ndarray] = {}
        self._fit_evt_total = 0
        self._fit_evt_valid = 0

        self._replay_draining = False
        self._replay_empty_streak = 0

    def set_trackfit_rt(self, rt: Any, *, n_lut: int = 2048, min_hits: int = 6) -> None:
        """
        Configure shared TrackFit instance used for continuous spectra filling.
        If rt is None or TrackFit init fails, fit-based spectra remain disabled.
        """
        if rt is None:
            self._trackfit = None
            self._qc_cache.clear()
            self._fit_cache.clear()
            self._reset_fit_hists()
            print("[Backend] TrackFit disabled (rt is None).")
            return
        try:
            self._trackfit = TrackFit.from_rt(
                rt,
                n_lut=int(n_lut),
                min_hits=int(min_hits),
                fit_only_kept_hits=True,
            )
            try:
                rt_tmax = float(getattr(rt, "Tmax"))
                if np.isfinite(rt_tmax) and rt_tmax > 0.0:
                    # Keep quality time window consistent with RT domain so fitted
                    # t_real can stay inside [T0, Tmax] without post-bounding.
                    self._quality.max_window_ns = rt_tmax
            except Exception:
                pass
            self._qc_cache.clear()
            self._fit_cache.clear()
            self._reset_fit_hists()
            print("[Backend] TrackFit initialized for live spectra.")
        except Exception as e:
            self._trackfit = None
            self._qc_cache.clear()
            self._fit_cache.clear()
            self._reset_fit_hists()
            print(f"[Backend] TrackFit init failed: {e}")

    def _reset_fit_hists(self) -> None:
        self._fit_adc_hist.fill(0)
        self._fit_adc_ch_hist.fill(0)
        self._fit_tdc_hist.fill(0)
        self._fit_tdc_ch_hist.fill(0)
        self._fit_resid_hist.fill(0)
        self._fit_resid_hist_by_chamber.clear()
        self._fit_evt_total = 0
        self._fit_evt_valid = 0

    @staticmethod
    def _fit_algorithm(fit: Any) -> str:
        try:
            return str(getattr(fit, "values", {}).get("algorithm", "") or "")
        except Exception:
            return ""

    @classmethod
    def _fit_is_accepted(cls, fit: Any, *, min_fit_hits: int, min_fit_dof: int) -> bool:
        if fit is None or not bool(getattr(fit, "ok", False)):
            return False

        # theta_scan_fallback is a diagnostic/candidate fit after the primary
        # triggerless fit failed. Keep its values available, but do not count it
        # as an accepted physics event.
        if cls._fit_algorithm(fit) == "theta_scan_fallback":
            return False

        try:
            fit_dof = int(round(float(getattr(fit, "values", {}).get("dof", fit.n_used - 3))))
        except Exception:
            fit_dof = int(getattr(fit, "n_used", 0)) - 3

        return bool((int(getattr(fit, "n_used", 0)) >= int(min_fit_hits)) and (fit_dof >= int(min_fit_dof)))

    def reset_fit_spectra(self) -> None:
        """
        Clear accumulated fit-based histograms. Geometry edits during replay do not
        retroactively remap old decoded data, so callers should clear these before
        rerunning replay with a new geometry.
        """
        self._reset_fit_hists()

    # ---------------- geometry registry (GUI-side) ----------------

    def _apply_quality_geometry(self) -> None:
        """
        Derive geometry-dependent QA cuts from the registered detector geometry.
        A Geometry object may override either value with attributes:
          - quality_max_adjacent_layer_x_jump_mm
          - quality_max_same_layer_x_span_mm
        """
        geos = [g for g in self._geos.values() if g is not None]
        if not geos:
            return

        geo = geos[0]
        col_pitch = float(getattr(geo, "column_distance", 0.0))
        if not (np.isfinite(col_pitch) and col_pitch > 0.0):
            return

        adj_jump = getattr(geo, "quality_max_adjacent_layer_x_jump_mm", None)
        same_span = getattr(geo, "quality_max_same_layer_x_span_mm", None)

        same_span_value = float(same_span) if same_span is not None else 2.0 * col_pitch
        self._quality.max_same_layer_x_span_mm = same_span_value
        self._quality.max_adjacent_layer_x_jump_mm = (
            float(adj_jump) if adj_jump is not None else same_span_value + 10.0
        )
        self._quality.n_layers = int(getattr(geo, "MAX_TUBE_LAYER", self._quality.n_layers))
        self._qc_cache.clear()
        self._fit_cache.clear()

    def set_geometries_from_list(self, geos: List[Any]) -> None:
        out: Dict[int, Any] = {}
        for g in (geos or []):
            if g is None:
                continue
            cid = int(getattr(g, "chamber_id", -1))
            if cid < 0:
                raise ValueError("Geometry missing valid chamber_id (must be >= 0)")
            if cid in out:
                raise ValueError(f"Duplicate chamber_id {cid}")
            out[cid] = g
        self._geos = out
        self._apply_quality_geometry()

    def set_geometries(self, geos: Dict[int, Any]) -> None:
        out: Dict[int, Any] = {}
        for cid, g in (geos or {}).items():
            if g is None:
                continue
            out[int(cid)] = g
        self._geos = out
        self._apply_quality_geometry()

    def get_geometry(self, chamber_id: int) -> Optional[Any]:
        return self._geos.get(int(chamber_id))

    def geometry_count(self) -> int:
        return len(self._geos)

    def geometries(self) -> Dict[int, Any]:
        return dict(self._geos)

    def _geo_for_decode_thread(self) -> Optional[Any]:
        """
        Always return a GeoRouter if at least one geometry exists.
        0 geos  -> None
        >=1     -> GeoRouter(tdcid -> geo)
        """
        geos = list(self._geos.values())
        if not geos:
            return None

        by_tdc: Dict[int, Any] = {}

        for g in geos:
            # ensure mapping arrays are applied
            try:
                g.set_assignment(g.slots_per_ml, g.ml0, g.ml1, apply_map=True)
            except Exception:
                pass

            max_tdc = int(getattr(g, "MAX_TDC", 40))
            is_active = getattr(g, "isActiveTDC", None)
            if not is_active:
                continue

            for t in range(max_tdc):
                try:
                    if int(is_active[t]) == 1:
                        if t in by_tdc:
                            print(f"[WARN] TDC {t} assigned to multiple chambers. First wins.")
                            continue
                        by_tdc[t] = g
                except Exception:
                    pass

        if not by_tdc:
            print("[WARN] No active TDCs found in any geometry.")
            return None

        print("[Decode geo router] TDC mapping:", sorted((t, g.chamber_id) for t, g in by_tdc.items()))
        return GeoRouter(by_tdc)

    # ---------------- public API ----------------

    def is_running(self) -> bool:
        return any(
            t is not None and t.isRunning()
            for t in (self._cap_thread, self._replay_thread, self._dec_thread)
        )

    def make_out_path(self, out_dir: str, run: int) -> str:
        out_dir = out_dir.strip() or os.getcwd()
        ts = _timestamp_yyyymmdd_hhmmss()
        fname = f"run{run:05d}_{ts}.dat"
        return os.path.join(out_dir, fname)

    def start_capture(self, dev: str, bpf: str, out_path: str, max_events_in_ram: int = 256):
        """
        Start capture + decode.
        DecodeThread is SINGLE and does NOT use geometry.
        """
        if self.is_running():
            self.stop_capture()

        self._current_out_path = str(out_path)

        # clear analysis queue (drop old bytes)
        try:
            while True:
                self._analysis_q.get_nowait()
        except queue.Empty:
            pass

        # reset navigation + caches on new run
        self.clear_event_cache(emit_signal=False)
        self._reset_fit_hists()

        # event buffer for valid events
        self._event_buf = EventBuffer(max_events=max_events_in_ram)

        # decode thread (consumer)
        geo_for_decode = self._geo_for_decode_thread()
        self._dec_thread = DecodeThread(
            analysis_q=self._analysis_q,
            event_buffer=self._event_buf,
            geo=geo_for_decode,
            max_tdcs=40,
            adc_bins=256,
            tdc_bins=4096,
            tdc_shift=5,
        )
        self._dec_thread.analysis_1hz.connect(self._on_decode_1hz)
        self._dec_thread.event_ready.connect(self._on_decoded_event)

        # capture thread (producer)
        self._cap_thread = CaptureThread(
            dev=dev,
            bpf=bpf,
            out_path=out_path,
            analysis_q=self._analysis_q,
        )
        self._cap_thread.message.connect(self.message)
        self._cap_thread.stats.connect(self.stats)
        self._cap_thread.finished.connect(self._on_source_finished)

        # start consumer first, then producer
        self._dec_thread.start()
        self.capture_started.emit()
        self._cap_thread.start()

        # broadcast "no current event" at run start
        self.event_changed.emit((None, None))

    def start_replay(self, dat_path: str, max_events_in_ram: int = 256, chunk_bytes: int = 1024 * 256):
        """
        Start offline decode/analysis from an existing .dat file.
        Reuses the same DecodeThread + analysis flow as live capture.
        """
        if self.is_running():
            self.stop_capture()

        self._current_out_path = str(dat_path)

        # clear analysis queue (drop old bytes)
        try:
            while True:
                self._analysis_q.get_nowait()
        except queue.Empty:
            pass

        # reset navigation + caches on new run
        self.clear_event_cache(emit_signal=False)
        self._reset_fit_hists()

        # event buffer for valid events
        self._event_buf = EventBuffer(max_events=max_events_in_ram)

        # decode thread (consumer)
        geo_for_decode = self._geo_for_decode_thread()
        self._dec_thread = DecodeThread(
            analysis_q=self._analysis_q,
            event_buffer=self._event_buf,
            geo=geo_for_decode,
            max_tdcs=40,
            adc_bins=256,
            tdc_bins=4096,
            tdc_shift=5,
        )
        self._dec_thread.analysis_1hz.connect(self._on_decode_1hz)
        self._dec_thread.event_ready.connect(self._on_decoded_event)

        # replay thread (producer)
        self._replay_thread = DatReplayThread(
            dat_path=str(dat_path),
            analysis_q=self._analysis_q,
            chunk_bytes=int(chunk_bytes),
        )
        self._replay_thread.message.connect(self.message)
        self._replay_thread.stats.connect(self.stats)
        self._replay_thread.replay_done.connect(self._on_replay_done)
        self._replay_thread.finished.connect(self._on_source_finished)

        # start consumer first, then producer
        self._dec_thread.start()
        self.capture_started.emit()
        self._replay_thread.start()

        self.event_changed.emit((None, None))

    @QtCore.pyqtSlot(object)
    def _on_decoded_event(self, ev: Any):
        """
        Process every decoded event and fill fit-based TDC spectra for chambers
        that pass quality checks.
        """
        if ev is None or self._trackfit is None:
            return

        self._fit_evt_total += 1

        try:
            time_by, valid_by = self._compute_quality_by_chamber(ev)
            fit_by = self._fit_results_by_chamber(ev)
        except Exception:
            return

        chambers = getattr(ev, "chambers", None)
        if not isinstance(chambers, dict):
            return

        tmin = float(self._fit_tdc_tmin_ns)
        binw = float(self._fit_tdc_binw_ns)
        nbins = int(self._fit_tdc_bins)
        rmin = float(self._fit_resid_min_mm)
        rbinw = float(self._fit_resid_binw_mm)
        rnbins = int(self._fit_resid_bins)
        evt_fit_ok = False

        for cid_raw, block in chambers.items():
            cid = int(cid_raw)
            if not bool(valid_by.get(cid, False)):
                continue

            hits = getattr(block, "hits", ())
            if not hits:
                continue

            tw = time_by.get(cid)
            if tw is None:
                continue

            t_corr = getattr(tw, "t_corr_ns", None)
            keep = getattr(tw, "keep_mask", None)
            if t_corr is None or keep is None:
                continue

            t_corr_arr = np.asarray(t_corr, dtype=np.float64)
            keep_arr = np.asarray(keep, dtype=bool)
            if t_corr_arr.ndim != 1 or keep_arr.ndim != 1:
                continue
            if t_corr_arr.shape[0] != len(hits) or keep_arr.shape[0] != len(hits):
                continue

            fit = fit_by.get(cid)
            if fit is None or not fit.ok:
                continue

            vals = fit.values
            if self._fit_algorithm(fit) == "theta_scan_fallback":
                continue

            t0_ns = float(vals.get("t0_ns", 0.0))

            used_mask = np.asarray(vals.get("used_mask_input", keep_arr), dtype=bool)
            if used_mask.shape[0] != len(hits):
                used_mask = keep_arr

            final_mask = keep_arr & used_mask
            if not np.any(final_mask):
                continue
            evt_fit_ok = True
            chamber_resid_hist = self._fit_resid_hist_by_chamber.get(cid)
            if chamber_resid_hist is None:
                chamber_resid_hist = np.zeros((self._fit_max_tdcs, self._fit_resid_bins), dtype=np.uint32)
                self._fit_resid_hist_by_chamber[cid] = chamber_resid_hist

            # Requested definition:
            #   real drift time = t_corr_ns - fitted t0
            t_real = t_corr_arr - t0_ns
            ib = np.floor((t_real - tmin) / binw).astype(np.int32)
            time_ok = np.isfinite(t_real) & (ib >= 0) & (ib < nbins)

            # Residual definition (same model used in trackfit_cpp):
            #   residual_mm = |n.p - c| - r(t_corr - t0)
            rb = None
            resid_ok = None
            try:
                nx = float(vals.get("nx"))
                ny = float(vals.get("ny"))
                c = float(vals.get("c"))

                x_arr = np.asarray([float(getattr(h, "x", np.nan)) for h in hits], dtype=np.float64)
                y_arr = np.asarray([float(getattr(h, "y", np.nan)) for h in hits], dtype=np.float64)
                dist = np.abs(nx * x_arr + ny * y_arr - c)
                r_fit = np.asarray(self._trackfit.radius_from_time_ns(t_corr_arr - t0_ns), dtype=np.float64)
                resid = dist - r_fit

                rb = np.floor((resid - rmin) / rbinw).astype(np.int32)
                resid_ok = np.isfinite(resid) & (rb >= 0) & (rb < rnbins)
            except Exception:
                rb = None
                resid_ok = None

            for i, h in enumerate(hits):
                if not bool(final_mask[i]):
                    continue
                tdc = int(getattr(h, "tdcid", -1))
                ch = int(getattr(h, "ch", -1))
                if not (0 <= tdc < self._fit_max_tdcs):
                    continue
                w = int(getattr(h, "width", -1))
                if 0 <= w < self._fit_adc_bins:
                    self._fit_adc_hist[tdc, w] += 1
                    if 0 <= ch < self._fit_max_channels:
                        self._fit_adc_ch_hist[tdc, ch, w] += 1
                if bool(time_ok[i]):
                    b = int(ib[i])
                    self._fit_tdc_hist[tdc, b] += 1
                    if 0 <= ch < self._fit_max_channels:
                        self._fit_tdc_ch_hist[tdc, ch, b] += 1
                if resid_ok is not None and bool(resid_ok[i]):
                    br = int(rb[i])
                    self._fit_resid_hist[tdc, br] += 1
                    chamber_resid_hist[tdc, br] += 1

        if evt_fit_ok:
            self._fit_evt_valid += 1

    @QtCore.pyqtSlot(object)
    def _on_decode_1hz(self, snap: Any):
        """
        Emit spectra snapshot that keeps decode counters and ADC histograms,
        but replaces TDC histograms with fit-based real drift-time spectra.
        """
        out = SimpleNamespace(
            # ADC remains decoder-native
            adc_hist=snap.adc_hist,
            adc_ch_hist=snap.adc_ch_hist,
            adc_bins=int(getattr(snap, "adc_bins", 0)),
            ch_adc_bins=int(getattr(snap, "ch_adc_bins", 0)),
            # ADC from fitted-hit subset only
            fit_adc_hist=self._fit_adc_hist,
            fit_adc_ch_hist=self._fit_adc_ch_hist,
            fit_adc_bins=int(self._fit_adc_bins),
            fit_ch_adc_bins=int(self._fit_adc_bins),

            # TDC becomes fit-based drift-time spectra
            tdc_hist=self._fit_tdc_hist,
            tdc_ch_hist=self._fit_tdc_ch_hist,
            tdc_bins=int(self._fit_tdc_bins),
            ch_tdc_bins=int(self._fit_tdc_bins),

            # Residual spectra from fit result (per TDC)
            fit_residual_hist=self._fit_resid_hist,
            fit_residual_hist_by_chamber={
                int(cid): arr for cid, arr in self._fit_resid_hist_by_chamber.items()
            },
            fit_residual_bins=int(self._fit_resid_bins),
            fit_residual_min_mm=float(self._fit_resid_min_mm),
            fit_residual_max_mm=float(self._fit_resid_max_mm),
            fit_residual_binw_mm=float(self._fit_resid_binw_mm),
            fit_evt_total=int(self._fit_evt_total),
            fit_evt_valid=int(self._fit_evt_valid),
            fit_evt_valid_ratio=float(self._fit_evt_valid / self._fit_evt_total) if self._fit_evt_total > 0 else 0.0,

            # decode counters / errors
            headers=int(getattr(snap, "headers", 0)),
            events_fifoA=int(getattr(snap, "events_fifoA", 0)),
            events_fifoB=int(getattr(snap, "events_fifoB", 0)),
            trailers=int(getattr(snap, "trailers", 0)),
            triggers=int(getattr(snap, "triggers", 0)),
            hits_total=int(getattr(snap, "hits_total", 0)),
            overflow_cnt=getattr(snap, "overflow_cnt", None),
            decode_err_cnt=getattr(snap, "decode_err_cnt", None),
            err_event_id=int(getattr(snap, "err_event_id", 0)),
            err_hit_count=int(getattr(snap, "err_hit_count", 0)),
            err_missing_trailer=int(getattr(snap, "err_missing_trailer", 0)),
            err_missing_header=int(getattr(snap, "err_missing_header", 0)),
            abort_burst=int(getattr(snap, "abort_burst", 0)),
            abort_watchdog=int(getattr(snap, "abort_watchdog", 0)),
            abort_watchdog_types=tuple(getattr(snap, "abort_watchdog_types", ())),
            events_buffered=int(getattr(snap, "events_buffered", 0)),

            # metadata for future UI usage/debug
            tdc_hist_source="fit_drift_time_ns:t_corr_minus_t0",
            tdc_time_min_ns=float(self._fit_tdc_tmin_ns),
            tdc_time_max_ns=float(self._fit_tdc_tmax_ns),
            tdc_time_binw_ns=float(self._fit_tdc_binw_ns),
            tdc_hist_raw=getattr(snap, "tdc_hist", None),
            tdc_ch_hist_raw=getattr(snap, "tdc_ch_hist", None),
        )
        self.analysis_1hz.emit(out)

    def stop_capture(self):
        self._replay_draining = False

        # stop producer first
        if self._cap_thread:
            self._cap_thread.stop()
            self._cap_thread.wait(2000)
            self._cap_thread = None
        if self._replay_thread:
            self._replay_thread.stop()
            self._replay_thread.wait(5000)
            self._replay_thread = None

        # then stop consumer
        if self._dec_thread:
            self._dec_thread.stop()
            self._dec_thread.wait(2000)
            self._dec_thread = None

        self.run_finished.emit()

    @QtCore.pyqtSlot()
    def _on_source_finished(self):
        # Keep this lightweight. Replay completion is finalized in _on_replay_done.
        pass

    @QtCore.pyqtSlot()
    def _on_replay_done(self):
        """
        Replay producer reached EOF (or stopped). Wait for decode queue to drain,
        then stop decode thread and mark run finished.
        """
        self._replay_draining = True
        self._replay_empty_streak = 0
        self._poll_replay_drain()

    def _poll_replay_drain(self):
        if not self._replay_draining:
            return

        if self._dec_thread is None:
            self._replay_draining = False
            self._replay_thread = None
            self.run_finished.emit()
            return

        qsize = self._analysis_q.qsize()
        if qsize == 0:
            self._replay_empty_streak += 1
        else:
            self._replay_empty_streak = 0

        # Require a short stable empty period so we don't race on the last chunk.
        if self._replay_empty_streak >= 3:
            self._dec_thread.stop()
            self._dec_thread.wait()
            self._dec_thread = None
            self._replay_draining = False
            self._replay_thread = None
            self.message.emit("[Replay] Decode complete.\n")
            self.run_finished.emit()
            return

        QtCore.QTimer.singleShot(100, self._poll_replay_drain)

    # ---------------- legacy event access (kept) ----------------

    def pop_event(self):
        """Return next valid Event from buffer (or None)."""
        if self._event_buf is None:
            return None
        return self._event_buf.pop()

    def events_buffered(self) -> int:
        if self._event_buf is None:
            return 0
        return self._event_buf.size()

    def current_out_path(self) -> Optional[str]:
        return self._current_out_path

    # ---------------- quality computation (chamber-based) ----------------

    def _compute_quality_by_chamber(self, ev: Any) -> Tuple[TimeByChamber, ValidByChamber]:
        """
        Compute quality per chamber (no mixing).
        Cached by event_id20.
        Returns: (time_by_chamber, valid_by_chamber)
        """
        if ev is None:
            return {}, {}

        try:
            eid20 = int(getattr(ev, "event_id20"))
        except Exception:
            eid20 = -1

        cached = self._qc_cache.get(eid20)
        if cached is not None:
            return cached

        q = self._quality.compute_for_event(ev)
        time_by = q.time_by_chamber
        valid_by = dict(q.valid_by_chamber)
        fit_by: FitByChamber = {}

        # Align chamber_valid with the fitter in the same spirit as
        # testbeam-mini-chamber_reco: QA must pass, the fit must succeed,
        # and enough inlier hits must remain after the residual cut.
        # A few rejected hits are acceptable and will stay visible as red.
        if self._trackfit is not None:
            chambers = getattr(ev, "chambers", None)
            if isinstance(chambers, dict):
                min_fit_hits = max(3, int(getattr(self._trackfit, "min_hits", 3)))
                min_fit_dof = max(0, min_fit_hits - 3)  # NPARS = 3 here: theta, intercept, t0
                for cid_raw, block in chambers.items():
                    cid = int(cid_raw)
                    if not bool(valid_by.get(cid, False)):
                        continue

                    hits = getattr(block, "hits", ())
                    tw = time_by.get(cid)
                    if not hits or tw is None:
                        valid_by[cid] = False
                        continue

                    t_corr = getattr(tw, "t_corr_ns", None)
                    keep = getattr(tw, "keep_mask", None)
                    if t_corr is None or keep is None:
                        valid_by[cid] = False
                        continue

                    t_corr_arr = np.asarray(t_corr, dtype=np.float64)
                    keep_arr = np.asarray(keep, dtype=bool)
                    if (
                        t_corr_arr.ndim != 1
                        or keep_arr.ndim != 1
                        or t_corr_arr.shape[0] != len(hits)
                        or keep_arr.shape[0] != len(hits)
                    ):
                        valid_by[cid] = False
                        continue

                    fit = self._trackfit.fit_chamber_hits(
                        chamber_id=cid,
                        hits=hits,
                        t_corr_ns=t_corr_arr,
                        keep_mask=keep_arr,
                    )
                    fit_by[cid] = fit
                    if not fit.ok:
                        valid_by[cid] = False
                        continue

                    valid_by[cid] = self._fit_is_accepted(
                        fit,
                        min_fit_hits=min_fit_hits,
                        min_fit_dof=min_fit_dof,
                    )

        out = (time_by, valid_by)

        self._qc_cache[eid20] = out
        self._fit_cache[eid20] = fit_by
        return out

    def _fit_results_by_chamber(self, ev: Any) -> FitByChamber:
        """
        Return cached fit results for this event, computing quality+fits on-demand
        if needed. The backend reuses these fits for both validity and spectra.
        """
        if ev is None:
            return {}

        try:
            eid20 = int(getattr(ev, "event_id20"))
        except Exception:
            eid20 = -1

        cached = self._fit_cache.get(eid20)
        if cached is not None:
            return cached

        self._compute_quality_by_chamber(ev)
        return self._fit_cache.get(eid20, {})

    def _emit_current_payload(self) -> None:
        """
        Emit (Event_raw, (time_by_chamber, valid_by_chamber)) or (None,None).
        """
        ev = self.current_event_raw()
        if ev is None:
            self.event_changed.emit((None, None))
            return

        time_by, valid_by = self._compute_quality_by_chamber(ev)
        self.event_changed.emit((ev, (time_by, valid_by)))

    # ---------------- global event navigation ----------------

    def cache_size(self) -> int:
        return len(self._ev_cache)

    def current_index(self) -> int:
        return int(self._ev_idx)

    def current_event_raw(self) -> Optional[Any]:
        """Return raw Event from cache (or None)."""
        if 0 <= self._ev_idx < len(self._ev_cache):
            return self._ev_cache[self._ev_idx]
        return None

    def current_event(self) -> EventPayload:
        """
        Return the same thing we emit:
          (Event_raw, (time_by_chamber, valid_by_chamber)) or (None,None)
        """
        ev = self.current_event_raw()
        if ev is None:
            return (None, None)
        time_by, valid_by = self._compute_quality_by_chamber(ev)
        return (ev, (time_by, valid_by))

    def goto_event(self, idx: int) -> None:
        idx = int(idx)
        if not (0 <= idx < len(self._ev_cache)):
            return
        self._ev_idx = idx
        self._emit_current_payload()

    def next_event(self) -> None:
        """
        Advance by 1:
          - If next is already cached, move forward.
          - Else, pop from EventBuffer, append raw Event to cache, select it.
        """
        # cached next
        if 0 <= self._ev_idx < len(self._ev_cache) - 1:
            self._ev_idx += 1
            self._emit_current_payload()
            return

        # need a fresh event
        ev = self.pop_event()
        if ev is None:
            # no new event; re-emit current (so UI can refresh buffered count)
            self._emit_current_payload()
            return

        self._ev_cache.append(ev)
        self._ev_idx = len(self._ev_cache) - 1
        self._emit_current_payload()

    def prev_event(self) -> None:
        """
        Go back by 1 within cache (cannot go back into EventBuffer).
        """
        if self._ev_idx <= 0:
            self._emit_current_payload()
            return
        self._ev_idx -= 1
        self._emit_current_payload()

    def clear_event_cache(self, emit_signal: bool = True, clear_buffer: bool = False):
        self._ev_cache.clear()
        self._ev_idx = -1
        self._qc_cache.clear()
        self._fit_cache.clear()

        if clear_buffer and self._event_buf is not None:
            self._event_buf.clear()

        if emit_signal:
            self.event_changed.emit((None, None))
