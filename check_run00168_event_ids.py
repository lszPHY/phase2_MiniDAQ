from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DAT_PATH = Path("data/run00169_20260328_221152.dat")
WORD_SIZE = 5
DEFAULT_CHUNK_WORDS = 1 << 20
TOP4_HDR = 0xA
TOP4_TRL = 0xC
EVENT_ID_MASK = 0xFFFFF
ABORT_TAG = 0xB
LEGACY_ABORT_WORD = 0xAAAAAAAAAA
ABORT_SUBTYPE_SHIFT = 32
ABORT_SUBTYPE_MASK = 0xF
BURST_ABORT_SUBTYPE = 0xA
WATCHDOG_ABORT_SUBTYPE = 0xD
BURST_ABORT_MAGIC_SHIFT = 10
BURST_ABORT_MAGIC_MASK = (1 << 22) - 1
BURST_ABORT_MAGIC = 0x2AAAAA


@dataclass(frozen=True, slots=True)
class MismatchExample:
    header_word_index: int
    trailer_word_index: int
    header_event_id: int
    trailer_event_id: int


@dataclass(frozen=True, slots=True)
class AbortEventExample:
    header_word_index: int
    abort_word_index: int
    trailer_word_index: int | None
    abort_kind: str
    event_id: int
    hit_count: int
    trailer_hit_count: int | None
    close_reason: str


def ids_are_consecutive(prev_id: int, cur_id: int) -> bool:
    return cur_id == ((prev_id + 1) & EVENT_ID_MASK)


def decode_abort_kind(word: int) -> str | None:
    if word == LEGACY_ABORT_WORD:
        return "legacy"

    top4 = (word >> 36) & 0xF
    if top4 != ABORT_TAG:
        return None

    subtype = (word >> ABORT_SUBTYPE_SHIFT) & ABORT_SUBTYPE_MASK
    if (
        subtype == BURST_ABORT_SUBTYPE
        and ((word >> BURST_ABORT_MAGIC_SHIFT) & BURST_ABORT_MAGIC_MASK) == BURST_ABORT_MAGIC
    ):
        return "burst"
    if (
        subtype == WATCHDOG_ABORT_SUBTYPE
    ):
        return "watchdog"
    return None


def scan_event_id_issues(
    dat_path: Path,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    max_examples: int = 10,
    max_abort_examples: int = 10,
) -> dict[str, object]:
    """Stream a .dat file chunk-by-chunk and count header/trailer ID issues."""
    chunk_bytes = WORD_SIZE * int(chunk_words)

    header_count = 0
    trailer_count = 0
    mismatch_count = 0
    inconsecutive_header_count = 0
    missing_trailer_count = 0
    missing_header_count = 0
    abort_marker_count = 0
    legacy_abort_count = 0
    burst_abort_count = 0
    watchdog_abort_count = 0

    mismatch_examples: list[MismatchExample] = []
    abort_event_examples: list[AbortEventExample] = []
    abort_hit_hist: Counter[int] = Counter()

    abort_event_count = 0
    abort_event_hit_total = 0
    abort_event_hit_min: int | None = None
    abort_event_hit_max: int | None = None
    abort_event_trailer_hit_mismatch_count = 0

    last_header_id: int | None = None
    open_header_id: int | None = None
    open_header_word: int | None = None
    open_hit_count = 0
    open_has_abort = False
    open_abort_word: int | None = None
    open_abort_kind: str | None = None
    word_base = 0

    def count_hits_between(prefix_hits: np.ndarray, start: int, stop: int) -> int:
        if stop <= start:
            return 0
        total = int(prefix_hits[stop - 1])
        if start > 0:
            total -= int(prefix_hits[start - 1])
        return total

    def record_abort_event(
        *,
        trailer_word_index: int | None,
        trailer_hit_count: int | None,
        close_reason: str,
    ) -> None:
        nonlocal abort_event_count
        nonlocal abort_event_hit_total
        nonlocal abort_event_hit_min
        nonlocal abort_event_hit_max
        nonlocal abort_event_trailer_hit_mismatch_count

        if open_header_id is None or open_abort_word is None or open_abort_kind is None:
            return

        hit_count = int(open_hit_count)

        abort_event_count += 1
        abort_event_hit_total += hit_count
        abort_hit_hist[hit_count] += 1

        if abort_event_hit_min is None or hit_count < abort_event_hit_min:
            abort_event_hit_min = hit_count
        if abort_event_hit_max is None or hit_count > abort_event_hit_max:
            abort_event_hit_max = hit_count

        if trailer_hit_count is not None and trailer_hit_count != hit_count:
            abort_event_trailer_hit_mismatch_count += 1

        if len(abort_event_examples) < max_abort_examples and open_header_word is not None:
            abort_event_examples.append(
                AbortEventExample(
                    header_word_index=open_header_word,
                    abort_word_index=open_abort_word,
                    trailer_word_index=trailer_word_index,
                    abort_kind=open_abort_kind,
                    event_id=open_header_id,
                    hit_count=hit_count,
                    trailer_hit_count=trailer_hit_count,
                    close_reason=close_reason,
                )
            )

    with dat_path.open("rb", buffering=chunk_bytes) as fh:
        leftover = b""

        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break

            if leftover:
                chunk = leftover + chunk

            whole = len(chunk) - (len(chunk) % WORD_SIZE)
            leftover = chunk[whole:]
            if whole == 0:
                continue

            words = np.frombuffer(chunk[:whole], dtype=np.uint8).reshape(-1, WORD_SIZE)
            w = (
                (words[:, 0].astype(np.uint64) << 32)
                | (words[:, 1].astype(np.uint64) << 24)
                | (words[:, 2].astype(np.uint64) << 16)
                | (words[:, 3].astype(np.uint64) << 8)
                | words[:, 4].astype(np.uint64)
            )

            top4 = (w >> 36) & 0xF
            is_legacy_abort = w == LEGACY_ABORT_WORD
            abort_subtype = (w >> ABORT_SUBTYPE_SHIFT) & ABORT_SUBTYPE_MASK
            burst_abort_magic = (w >> BURST_ABORT_MAGIC_SHIFT) & BURST_ABORT_MAGIC_MASK
            is_burst_abort = (
                (top4 == ABORT_TAG)
                & (abort_subtype == BURST_ABORT_SUBTYPE)
                & (burst_abort_magic == BURST_ABORT_MAGIC)
            )
            is_watchdog_abort = (
                (top4 == ABORT_TAG)
                & (abort_subtype == WATCHDOG_ABORT_SUBTYPE)
            )
            is_abort = is_legacy_abort | is_burst_abort | is_watchdog_abort
            is_trigger = (top4 == 0xE) & (((w >> 30) & 0x3F) == 0)
            is_decode_error = (w & 0xFFFFFFFF) == 0xF7411111
            is_overflow = ((w >> 24) & 0xFF) == 0xE8
            is_hit = ~(is_abort | (top4 == TOP4_HDR) | (top4 == TOP4_TRL) | is_trigger | is_decode_error | is_overflow)

            abort_marker_count += int(np.count_nonzero(is_abort))
            legacy_abort_count += int(np.count_nonzero(is_legacy_abort))
            burst_abort_count += int(np.count_nonzero(is_burst_abort))
            watchdog_abort_count += int(np.count_nonzero(is_watchdog_abort))

            header_idx = np.flatnonzero((top4 == TOP4_HDR) & ~is_abort).astype(np.int64)
            trailer_idx = np.flatnonzero(top4 == TOP4_TRL).astype(np.int64)
            abort_idx = np.flatnonzero(is_abort).astype(np.int64)
            header_ids = ((w[header_idx] >> 16) & EVENT_ID_MASK).astype(np.int64)
            trailer_ids = ((w[trailer_idx] >> 16) & EVENT_ID_MASK).astype(np.int64)
            trailer_hit_counts = (w[trailer_idx] & 0x3FF).astype(np.int64)
            prefix_hits = np.cumsum(is_hit, dtype=np.int64)

            header_count += int(header_ids.size)
            trailer_count += int(trailer_ids.size)

            if header_ids.size:
                if last_header_id is not None and not ids_are_consecutive(last_header_id, int(header_ids[0])):
                    inconsecutive_header_count += 1

                if header_ids.size >= 2:
                    expected = (header_ids[:-1] + 1) & EVENT_ID_MASK
                    inconsecutive_header_count += int(np.count_nonzero(header_ids[1:] != expected))

                last_header_id = int(header_ids[-1])

            hi = 0
            ti = 0
            ai = 0
            cursor = 0

            while hi < header_idx.size or ti < trailer_idx.size or ai < abort_idx.size:
                next_header = int(header_idx[hi]) if hi < header_idx.size else None
                next_trailer = int(trailer_idx[ti]) if ti < trailer_idx.size else None
                next_abort = int(abort_idx[ai]) if ai < abort_idx.size else None

                next_pos = None
                next_kind = ""

                if next_header is not None:
                    next_pos = next_header
                    next_kind = "header"
                if next_trailer is not None and (next_pos is None or next_trailer < next_pos):
                    next_pos = next_trailer
                    next_kind = "trailer"
                if next_abort is not None and (next_pos is None or next_abort < next_pos):
                    next_pos = next_abort
                    next_kind = "abort"

                if next_pos is None:
                    break

                if open_header_id is not None:
                    open_hit_count += count_hits_between(prefix_hits, cursor, next_pos)
                cursor = next_pos + 1

                if next_kind == "header":
                    if open_header_id is not None:
                        record_abort_event(
                            trailer_word_index=None,
                            trailer_hit_count=None,
                            close_reason="missing_trailer",
                        )
                        missing_trailer_count += 1
                    open_header_id = int(header_ids[hi])
                    open_header_word = word_base + int(header_idx[hi])
                    open_hit_count = 0
                    open_has_abort = False
                    open_abort_word = None
                    open_abort_kind = None
                    hi += 1
                    continue

                if next_kind == "abort":
                    if open_header_id is not None:
                        abort_pos = int(abort_idx[ai])
                        abort_word = int(w[abort_pos])
                        abort_kind = decode_abort_kind(abort_word) or "legacy"
                        open_has_abort = True
                        open_abort_word = word_base + abort_pos
                        open_abort_kind = abort_kind
                    ai += 1
                    continue

                trailer_word = word_base + int(trailer_idx[ti])
                trailer_id = int(trailer_ids[ti])
                trailer_hit_count = int(trailer_hit_counts[ti])
                if open_header_id is None:
                    missing_header_count += 1
                else:
                    if open_header_id != trailer_id:
                        mismatch_count += 1
                        if len(mismatch_examples) < max_examples and open_header_word is not None:
                            mismatch_examples.append(
                                MismatchExample(
                                    header_word_index=open_header_word,
                                    trailer_word_index=trailer_word,
                                    header_event_id=open_header_id,
                                    trailer_event_id=trailer_id,
                                )
                            )
                    if open_has_abort:
                        record_abort_event(
                            trailer_word_index=trailer_word,
                            trailer_hit_count=trailer_hit_count,
                            close_reason="trailer",
                        )
                    open_header_id = None
                    open_header_word = None
                    open_hit_count = 0
                    open_has_abort = False
                    open_abort_word = None
                    open_abort_kind = None

                ti += 1

            if open_header_id is not None:
                open_hit_count += count_hits_between(prefix_hits, cursor, int(words.shape[0]))

            word_base += int(words.shape[0])

    if open_header_id is not None:
        if open_has_abort:
            record_abort_event(
                trailer_word_index=None,
                trailer_hit_count=None,
                close_reason="eof",
            )
        missing_trailer_count += 1

    abort_hit_hist_top = sorted(
        abort_hit_hist.items(),
        key=lambda item: (-item[1], item[0]),
    )
    abort_hit_mean = (abort_event_hit_total / abort_event_count) if abort_event_count else None

    return {
        "header_count": header_count,
        "trailer_count": trailer_count,
        "header_trailer_id_mismatch_count": mismatch_count,
        "inconsecutive_header_id_count": inconsecutive_header_count,
        "missing_trailer_like_gui_count": missing_trailer_count,
        "missing_header_like_gui_count": missing_header_count,
        "abort_count": abort_marker_count,
        "abort_marker_count": abort_marker_count,
        "legacy_abort_count": legacy_abort_count,
        "burst_abort_count": burst_abort_count,
        "watchdog_abort_count": watchdog_abort_count,
        "abort_event_count": abort_event_count,
        "abort_event_hit_total": abort_event_hit_total,
        "abort_event_hit_count_min": abort_event_hit_min,
        "abort_event_hit_count_max": abort_event_hit_max,
        "abort_event_hit_count_mean": abort_hit_mean,
        "abort_event_trailer_hit_count_mismatch_count": abort_event_trailer_hit_mismatch_count,
        "abort_event_hit_hist_top": abort_hit_hist_top,
        "abort_event_examples": abort_event_examples,
        "trailing_bytes_ignored": len(leftover),
        "mismatch_examples": mismatch_examples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count event-header / event-trailer ID mismatches in a MiniDAQ .dat file."
    )
    parser.add_argument(
        "dat_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DAT_PATH,
        help=f".dat file to scan (default: {DEFAULT_DAT_PATH})",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=DEFAULT_CHUNK_WORDS,
        help="number of 5-byte words to process per chunk",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="maximum number of mismatch examples to print",
    )
    parser.add_argument(
        "--abort-examples",
        type=int,
        default=10,
        help="maximum number of abort-event examples to print",
    )
    parser.add_argument(
        "--abort-hit-top",
        type=int,
        default=10,
        help="maximum number of most-common abort-event hit counts to print",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dat_path = args.dat_path

    print(f"reading {dat_path}")
    print(f"size_bytes={dat_path.stat().st_size}")

    stats = scan_event_id_issues(
        dat_path,
        chunk_words=args.chunk_words,
        max_examples=max(0, int(args.examples)),
        max_abort_examples=max(0, int(args.abort_examples)),
    )

    print(f"header_count={stats['header_count']}")
    print(f"trailer_count={stats['trailer_count']}")
    print(f"header_trailer_id_mismatch_count={stats['header_trailer_id_mismatch_count']}")
    print(f"missing_trailer_like_gui_count={stats['missing_trailer_like_gui_count']}")
    print(f"missing_header_like_gui_count={stats['missing_header_like_gui_count']}")
    print(f"inconsecutive_header_id_count={stats['inconsecutive_header_id_count']}")
    print(f"abort_count={stats['abort_count']}")
    print(f"abort_marker_count={stats['abort_marker_count']}")
    print(f"legacy_abort_count={stats['legacy_abort_count']}")
    print(f"burst_abort_count={stats['burst_abort_count']}")
    print(f"watchdog_abort_count={stats['watchdog_abort_count']}")
    print(f"abort_event_count={stats['abort_event_count']}")
    print(f"abort_event_hit_total={stats['abort_event_hit_total']}")
    print(f"abort_event_hit_count_min={stats['abort_event_hit_count_min']}")
    print(f"abort_event_hit_count_max={stats['abort_event_hit_count_max']}")
    print(f"abort_event_hit_count_mean={stats['abort_event_hit_count_mean']}")
    print(
        "abort_event_trailer_hit_count_mismatch_count="
        f"{stats['abort_event_trailer_hit_count_mismatch_count']}"
    )
    print(f"trailing_bytes_ignored={stats['trailing_bytes_ignored']}")

    examples = stats["mismatch_examples"]
    if examples:
        for idx, example in enumerate(examples, start=1):
            print(
                "mismatch_example"
                f"[{idx}]"
                f" header_word_index={example.header_word_index}"
                f" trailer_word_index={example.trailer_word_index}"
                f" header_event_id={example.header_event_id}"
                f" trailer_event_id={example.trailer_event_id}"
                f" header_event_id_hex=0x{example.header_event_id:05x}"
                f" trailer_event_id_hex=0x{example.trailer_event_id:05x}"
            )
    else:
        print("mismatch_examples=none")

    abort_hit_hist_top = stats["abort_event_hit_hist_top"]
    if abort_hit_hist_top:
        for idx, (hit_count, event_count) in enumerate(
            abort_hit_hist_top[: max(0, int(args.abort_hit_top))],
            start=1,
        ):
            print(
                "abort_event_hit_count_top"
                f"[{idx}]"
                f" hit_count={hit_count}"
                f" event_count={event_count}"
            )
    else:
        print("abort_event_hit_count_top=none")

    abort_examples = stats["abort_event_examples"]
    if abort_examples:
        for idx, example in enumerate(abort_examples, start=1):
            print(
                "abort_event_example"
                f"[{idx}]"
                f" header_word_index={example.header_word_index}"
                f" abort_word_index={example.abort_word_index}"
                f" trailer_word_index={example.trailer_word_index}"
                f" abort_kind={example.abort_kind}"
                f" event_id={example.event_id}"
                f" event_id_hex=0x{example.event_id:05x}"
                f" hit_count={example.hit_count}"
                f" trailer_hit_count={example.trailer_hit_count}"
                f" close_reason={example.close_reason}"
            )
    else:
        print("abort_event_examples=none")


if __name__ == "__main__":
    main()
