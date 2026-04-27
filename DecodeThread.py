from __future__ import annotations

from collections import Counter
import time
import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt5 import QtCore

from Signal import SignalType, Hit, decode_stream, GeometryLike
from Event import Event, ChamberBlock  # UPDATED


_WATCHDOG_TYPE_ORDER = (
    "IDLE",
    "TRIG_ONLY",
    "START_EVENT",
    "CAPTURE_PRE",
    "SEND_HEAD",
    "FLUSH_BUF",
    "STREAM_LIVE",
    "SEND_TRAIL",
    "ABORT_MARK",
    "ABORT_TRAIL",
    "ABORT_CLEAN",
    "legacy/unknown",
)


@dataclass(frozen=True, slots=True)
class DecodeSnapshot:
    adc_hist: np.ndarray
    tdc_hist: np.ndarray

    adc_ch_hist: np.ndarray
    tdc_ch_hist: np.ndarray

    adc_bins: int
    tdc_bins: int
    ch_adc_bins: int
    ch_tdc_bins: int

    headers: int
    events_fifoA: int
    events_fifoB: int
    trailers: int
    triggers: int
    hits_total: int

    overflow_cnt: np.ndarray
    decode_err_cnt: np.ndarray

    err_event_id: int
    err_hit_count: int
    err_missing_trailer: int
    err_missing_header: int
    abort_burst: int
    abort_watchdog: int
    abort_watchdog_types: tuple[tuple[str, int], ...]

    events_buffered: int


class EventBuffer:
    def __init__(self, max_events: int = 256):
        self._q: "queue.Queue[Event]" = queue.Queue(maxsize=max_events)
        self.dropped = 0

    def push(self, ev: Event):
        try:
            self._q.put_nowait(ev)
        except queue.Full:
            self.dropped += 1

    def pop(self) -> Optional[Event]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def size(self) -> int:
        return self._q.qsize()

    def clear(self):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass


class DecodeThread(QtCore.QThread):
    analysis_1hz = QtCore.pyqtSignal(object)
    event_ready = QtCore.pyqtSignal(object)

    def __init__(
        self,
        analysis_q: "queue.Queue[bytes]",
        event_buffer: EventBuffer,
        geo: Optional["GeometryLike"] = None,
        max_tdcs: int = 40,
        max_channels: int = 24,
        adc_bins: int = 256,
        tdc_bins: int = 4096,
        tdc_shift: int = 5,
        ch_tdc_bins: int = 1024,
        ch_tdc_shift: int = 7,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)

        self.q = analysis_q
        self.buf = event_buffer
        self.geo = geo
        self._stop = False

        self.max_tdcs = int(max_tdcs)
        self.max_channels = int(max_channels)

        self.adc_bins = int(adc_bins)
        self.tdc_bins = int(tdc_bins)
        self.tdc_shift = int(tdc_shift)

        self.ch_adc_bins = int(adc_bins)
        self.ch_tdc_bins = int(ch_tdc_bins)
        self.ch_tdc_shift = int(ch_tdc_shift)

        # histograms (VALID EVENTS ONLY)
        self._adc = np.zeros((self.max_tdcs, self.adc_bins), dtype=np.uint32)
        self._tdc = np.zeros((self.max_tdcs, self.tdc_bins), dtype=np.uint32)

        self._adc_ch = np.zeros((self.max_tdcs, self.max_channels, self.ch_adc_bins), dtype=np.uint32)
        self._tdc_ch = np.zeros((self.max_tdcs, self.max_channels, self.ch_tdc_bins), dtype=np.uint32)

        self._ovf = np.zeros((self.max_tdcs,), dtype=np.uint32)
        self._derr = np.zeros((self.max_tdcs,), dtype=np.uint32)

        self._hdr = 0
        self._events_fifoA = 0
        self._events_fifoB = 0
        self._trl = 0
        self._trg = 0
        self._hit_total = 0

        self._err_eid = 0
        self._err_hit = 0
        self._err_missing_trailer = 0
        self._err_missing_header = 0
        self._abort_burst = 0
        self._abort_watchdog = 0
        self._abort_watchdog_types: Counter[str] = Counter()

        # current event state
        self._cur_open = False
        self._cur_event_id20 = 0
        self._cur_rd_bank_sel = 0

        # Keep for histogram updates (fast flat loop)
        self._cur_hits: list[Hit] = []

        # Chamber grouping for Event payload (no mixing)
        self._cur_hits_by_ch: dict[int, list[Hit]] = {}

        self._cur_hit_count = 0
        self._last_emit = time.time()

    def stop(self):
        self._stop = True

    def _reset_event(self):
        self._cur_open = False
        self._cur_event_id20 = 0
        self._cur_rd_bank_sel = 0
        self._cur_hits.clear()
        self._cur_hits_by_ch.clear()
        self._cur_hit_count = 0

    def _start_event(self, event_id20: int, rd_bank_sel: int):
        self._cur_open = True
        self._cur_event_id20 = int(event_id20)
        self._cur_rd_bank_sel = int(rd_bank_sel)
        self._cur_hits.clear()
        self._cur_hits_by_ch.clear()
        self._cur_hit_count = 0

    @staticmethod
    def _watchdog_type_name(abort_marker) -> str:
        state_name = str(getattr(abort_marker, "watchdog_state_name", "") or "").strip()
        if state_name:
            return state_name

        state_bits = getattr(abort_marker, "watchdog_state_bits", None)
        if state_bits is None:
            return "legacy/unknown"

        return f"state 0x{int(state_bits):03x}"

    def _ordered_watchdog_types(self) -> tuple[tuple[str, int], ...]:
        order = {name: idx for idx, name in enumerate(_WATCHDOG_TYPE_ORDER)}
        items = sorted(
            self._abort_watchdog_types.items(),
            key=lambda item: (order.get(item[0], len(order)), item[0]),
        )
        return tuple((str(name), int(count)) for name, count in items if int(count) > 0)

    def _finalize_event(self, trailer_event_id20: int, trigger_count: int, hit_expected: int):
        if self._cur_event_id20 != int(trailer_event_id20):
            self._err_eid += 1
            self._reset_event()
            return

        if self._cur_hit_count != int(hit_expected):
            self._err_hit += 1
            self._reset_event()
            return

        # VALID: update histograms (keep locals for speed)
        adc = self._adc
        tdc = self._tdc
        adc_ch = self._adc_ch
        tdc_ch = self._tdc_ch

        max_tdcs = self.max_tdcs
        max_channels = self.max_channels
        adc_bins = self.adc_bins
        tdc_bins = self.tdc_bins
        tdc_shift = self.tdc_shift
        ch_adc_bins = self.ch_adc_bins
        ch_tdc_bins = self.ch_tdc_bins
        ch_tdc_shift = self.ch_tdc_shift

        for h in self._cur_hits:
            t = int(h.tdcid)
            if not (0 <= t < max_tdcs):
                continue

            ch = int(h.ch)
            w = int(h.width)

            if 0 <= w < adc_bins:
                adc[t, w] += 1

            b = int(h.ledge) >> tdc_shift
            if b < 0:
                b = 0
            elif b >= tdc_bins:
                b = tdc_bins - 1
            tdc[t, b] += 1

            if 0 <= ch < max_channels:
                if 0 <= w < ch_adc_bins:
                    adc_ch[t, ch, w] += 1

                bc = int(h.ledge) >> ch_tdc_shift
                if bc < 0:
                    bc = 0
                elif bc >= ch_tdc_bins:
                    bc = ch_tdc_bins - 1
                tdc_ch[t, ch, bc] += 1

        # Freeze chamber-separated hits into ChamberBlocks
        chambers = {
            int(cid): ChamberBlock(chamber_id=int(cid), hits=tuple(hlist))
            for cid, hlist in self._cur_hits_by_ch.items()
            if hlist
        }

        ev = Event(
            event_id20=int(self._cur_event_id20),
            rd_bank_sel=int(self._cur_rd_bank_sel),
            trigger_count=int(trigger_count),
            hit_count_expected=int(hit_expected),
            chambers=chambers,
        )
        self.buf.push(ev)
        self.event_ready.emit(ev)
        self._reset_event()

    def run(self):
        q_get = self.q.get
        geo = self.geo
        emit_1hz = self._emit_1hz_if_needed

        while not self._stop:
            try:
                chunk = q_get(timeout=0.01)
            except queue.Empty:
                emit_1hz()
                continue

            for s in decode_stream(chunk, geo=geo):
                st = s.type

                if st == SignalType.EVENT_HEADER and s.header is not None:
                    self._hdr += 1
                    if int(s.header.rd_bank_sel) == 0:
                        self._events_fifoA += 1
                    else:
                        self._events_fifoB += 1
                    if self._cur_open:
                        self._err_missing_trailer += 1
                        self._reset_event()
                    self._start_event(s.header.event_id20, s.header.rd_bank_sel)
                    continue

                if st == SignalType.HIT and s.hit is not None:
                    self._hit_total += 1
                    if self._cur_open:
                        self._cur_hits.append(s.hit)
                        self._cur_hit_count += 1

                        cid = int(getattr(s.hit, "chamber_id", -1))
                        if cid >= 0:
                            self._cur_hits_by_ch.setdefault(cid, []).append(s.hit)
                    continue

                if st == SignalType.EVENT_TRAILER and s.trailer is not None:
                    self._trl += 1
                    if not self._cur_open:
                        self._err_missing_header += 1
                        continue

                    self._finalize_event(
                        trailer_event_id20=int(s.trailer.event_id20),
                        trigger_count=int(s.trailer.trigger_count),
                        hit_expected=int(s.trailer.hit_count),
                    )
                    continue

                if st == SignalType.TRIGGER:
                    self._trg += 1
                    continue

                if st == SignalType.ABORT and s.abort is not None:
                    abort_kind = str(getattr(s.abort, "kind", "burst"))
                    if abort_kind == "watchdog":
                        self._abort_watchdog += 1
                        watchdog_type = self._watchdog_type_name(s.abort)
                        self._abort_watchdog_types[watchdog_type] += 1
                    else:
                        self._abort_burst += 1
                    continue

                if st == SignalType.OVERFLOW and s.overflow is not None:
                    t = int(s.overflow.tdcid)
                    if 0 <= t < self.max_tdcs:
                        self._ovf[t] += 1
                    continue

                if st == SignalType.DECODE_ERROR and s.error is not None:
                    t = int(s.error.tdcid)
                    if 0 <= t < self.max_tdcs:
                        self._derr[t] += 1
                    continue

            emit_1hz()

    def _emit_1hz_if_needed(self):
        now = time.time()
        if now - self._last_emit < 1.0:
            return

        snap = DecodeSnapshot(
            adc_hist=self._adc,
            tdc_hist=self._tdc,
            adc_ch_hist=self._adc_ch,
            tdc_ch_hist=self._tdc_ch,
            adc_bins=self.adc_bins,
            tdc_bins=self.tdc_bins,
            ch_adc_bins=self.ch_adc_bins,
            ch_tdc_bins=self.ch_tdc_bins,
            headers=self._hdr,
            events_fifoA=self._events_fifoA,
            events_fifoB=self._events_fifoB,
            trailers=self._trl,
            triggers=self._trg,
            hits_total=self._hit_total,
            overflow_cnt=self._ovf,
            decode_err_cnt=self._derr,
            err_event_id=self._err_eid,
            err_hit_count=self._err_hit,
            err_missing_trailer=self._err_missing_trailer,
            err_missing_header=self._err_missing_header,
            abort_burst=self._abort_burst,
            abort_watchdog=self._abort_watchdog,
            abort_watchdog_types=self._ordered_watchdog_types(),
            events_buffered=self.buf.size(),
        )
        self.analysis_1hz.emit(snap)
        self._last_emit = now
