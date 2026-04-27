# Signal.py
# MiniDAQ signal definitions + decoder
# Hit includes spatial info; Overflow / DecodeError include tdcid

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Iterator, Protocol, Tuple

WORD_SIZE = 5


class SignalType(IntEnum):
    HIT = 1
    TRIGGER = 2
    EVENT_HEADER = 3
    EVENT_TRAILER = 4
    OVERFLOW = 5
    DECODE_ERROR = 6
    ABORT = 7
    UNKNOWN = 99


# ---- geometry hook ----

class GeometryLike(Protocol):
    def wire_center_from_hit(
        self, tdc_id: int, channel_id: int
    ) -> Tuple[float, float, int, int]:
        ...


# ---------------- payloads ----------------

@dataclass(frozen=True, slots=True)
class Hit:
    csmid: int
    tdcid: int
    ch: int
    mode: int
    ledge: int
    width: int

    x: float
    y: float
    layer: int
    col: int
    chamber_id: int


@dataclass(frozen=True, slots=True)
class EventHeader:
    event_id20: int
    rd_bank_sel: int


@dataclass(frozen=True, slots=True)
class EventTrailer:
    event_id20: int
    trigger_count: int
    hit_count: int


@dataclass(frozen=True, slots=True)
class Trigger:
    raw: int
    overflow: int
    event_count: int
    ledge: int


@dataclass(frozen=True, slots=True)
class Overflow:
    tdcid: int
    raw: int


@dataclass(frozen=True, slots=True)
class DecodeError:
    tdcid: int
    raw: int


@dataclass(frozen=True, slots=True)
class AbortMarker:
    raw: int
    kind: str
    threshold: Optional[int] = None
    watchdog_count: Optional[int] = None
    watchdog_state_bits: Optional[int] = None
    watchdog_state_name: Optional[str] = None


# ---------------- Signal ----------------

@dataclass(frozen=True, slots=True)
class Signal:
    type: SignalType
    raw40: int

    hit: Optional[Hit] = None
    trigger: Optional[Trigger] = None
    header: Optional[EventHeader] = None
    trailer: Optional[EventTrailer] = None
    overflow: Optional[Overflow] = None
    error: Optional[DecodeError] = None
    abort: Optional[AbortMarker] = None


# ---------------- decoder ----------------

_HDR = 0xA
_TRL = 0xC
_TRG = 0xE
_OVF_MAGIC = 0xE8
_ERR_MAGIC = 0xF7411111
_ABORT_TAG = 0xB
_ABORT_SUBTYPE_SHIFT = 32
_ABORT_SUBTYPE_MASK = 0xF
_BURST_ABORT_SUBTYPE = 0xA
_WATCHDOG_ABORT_SUBTYPE = 0xD
_BURST_ABORT_MAGIC_SHIFT = 10
_BURST_ABORT_MAGIC_MASK = (1 << 22) - 1
_BURST_ABORT_MAGIC = 0x2AAAAA
_BURST_ABORT_THRESHOLD_MASK = (1 << 10) - 1
_WATCHDOG_ABORT_STATE_SHIFT = 21
_WATCHDOG_ABORT_STATE_MASK = (1 << 11) - 1
_WATCHDOG_ABORT_COUNT_MASK = (1 << 21) - 1
_WATCHDOG_ABORT_LEGACY_MAGIC = 0x6DB

_WATCHDOG_STATE_NAMES = {
    1 << 0: "IDLE",
    1 << 1: "TRIG_ONLY",
    1 << 2: "START_EVENT",
    1 << 3: "CAPTURE_PRE",
    1 << 4: "SEND_HEAD",
    1 << 5: "FLUSH_BUF",
    1 << 6: "STREAM_LIVE",
    1 << 7: "SEND_TRAIL",
    1 << 8: "ABORT_MARK",
    1 << 9: "ABORT_TRAIL",
    1 << 10: "ABORT_CLEAN",
}


def _decode_tdcid(w: int) -> int:
    """Shared TDC ID extraction."""
    csmid = (w >> 37) & 0x7
    tdc_local = (w >> 32) & 0x1F
    return tdc_local + csmid * 20


def _decode_watchdog_state_name(state_bits: int) -> Optional[str]:
    return _WATCHDOG_STATE_NAMES.get(state_bits)


def decode_word5(word5: bytes, geo: Optional[GeometryLike] = None) -> Signal:
    if len(word5) != WORD_SIZE:
        raise ValueError("invalid word size")

    w = int.from_bytes(word5, "big")
    top4 = (w >> 36) & 0xF

    # ---------- Abort marker ----------
    if top4 == _ABORT_TAG:
        abort_subtype = (w >> _ABORT_SUBTYPE_SHIFT) & _ABORT_SUBTYPE_MASK
        if (
            abort_subtype == _BURST_ABORT_SUBTYPE
            and ((w >> _BURST_ABORT_MAGIC_SHIFT) & _BURST_ABORT_MAGIC_MASK) == _BURST_ABORT_MAGIC
        ):
            return Signal(
                SignalType.ABORT,
                w,
                abort=AbortMarker(
                    raw=w,
                    kind="burst",
                    threshold=w & _BURST_ABORT_THRESHOLD_MASK,
                ),
            )

        if (
            abort_subtype == _WATCHDOG_ABORT_SUBTYPE
        ):
            watchdog_state_bits = (w >> _WATCHDOG_ABORT_STATE_SHIFT) & _WATCHDOG_ABORT_STATE_MASK
            watchdog_state_name = _decode_watchdog_state_name(watchdog_state_bits)
            if watchdog_state_bits == _WATCHDOG_ABORT_LEGACY_MAGIC:
                watchdog_state_bits = None
                watchdog_state_name = None
            return Signal(
                SignalType.ABORT,
                w,
                abort=AbortMarker(
                    raw=w,
                    kind="watchdog",
                    watchdog_count=w & _WATCHDOG_ABORT_COUNT_MASK,
                    watchdog_state_bits=watchdog_state_bits,
                    watchdog_state_name=watchdog_state_name,
                ),
            )

    # ---------- Event header ----------
    if top4 == _HDR:
        return Signal(
            SignalType.EVENT_HEADER,
            w,
            header=EventHeader(
                event_id20=(w >> 16) & 0xFFFFF,
                rd_bank_sel=(w >> 15) & 0x1,
            ),
        )

    # ---------- Event trailer ----------
    if top4 == _TRL:
        return Signal(
            SignalType.EVENT_TRAILER,
            w,
            trailer=EventTrailer(
                event_id20=(w >> 16) & 0xFFFFF,
                trigger_count=(w >> 10) & 0x3F,
                hit_count=w & 0x3FF,
            ),
        )

    # ---------- Trigger ----------
    if top4 == _TRG and ((w >> 30) & 0x3F) == 0:
        return Signal(
            SignalType.TRIGGER,
            w,
            trigger=Trigger(
                raw=w,
                overflow=(w >> 29) & 0x1,
                event_count=(w >> 17) & 0xFFF,
                ledge=w & 0x1FFFF,
            ),
        )

    low32 = w & 0xFFFFFFFF

    # ---------- Decode error ----------
    if low32 == _ERR_MAGIC:
        tdcid = _decode_tdcid(w)
        return Signal(
            SignalType.DECODE_ERROR,
            w,
            error=DecodeError(tdcid=tdcid, raw=w),
        )

    # ---------- Overflow ----------
    if ((w >> 24) & 0xFF) == _OVF_MAGIC:
        tdcid = _decode_tdcid(w)
        return Signal(
            SignalType.OVERFLOW,
            w,
            overflow=Overflow(tdcid=tdcid, raw=w),
        )

    # ---------- Hit ----------
    csmid = (w >> 37) & 0x7
    tdcid = _decode_tdcid(w)
    ch = (w >> 27) & 0x1F

    x = -1.0
    y = -1.0
    layer = -1
    col = -1
    cid = -1
    if geo is not None:
        try:
            x, y, layer, col = geo.wire_center_from_hit(tdcid, ch)
        except Exception:
            pass
            
    if geo is not None and hasattr(geo, "chamber_id_from_tdcid"):
        try:
            cid = int(geo.chamber_id_from_tdcid(tdcid))
        except Exception:
            cid = -1

    return Signal(
        SignalType.HIT,
        w,
        hit=Hit(
            csmid=csmid,
            tdcid=tdcid,
            ch=ch,
            mode=(w >> 25) & 0x3,
            ledge=(w >> 8) & 0x1FFFF,
            width=w & 0xFF,
            x=float(x),
            y=float(y),
            layer=int(layer),
            col=int(col),
            chamber_id=cid,
        ),
    )


def decode_stream(buf: bytes, geo: Optional[GeometryLike] = None) -> Iterator[Signal]:
    n = len(buf) // WORD_SIZE
    for i in range(n):
        yield decode_word5(
            buf[i * WORD_SIZE : (i + 1) * WORD_SIZE],
            geo=geo,
        )
