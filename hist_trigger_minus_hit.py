from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DAT_PATH = Path("data/run00131_20260309_164054.dat")
WORD_SIZE = 5

TOP4_HDR = 0xA
TOP4_TRL = 0xC
TOP4_TRG = 0xE
LOW6_MASK = 0x3F
OVF_MAGIC = 0xE8
ERR_MAGIC = 0xF7411111

LEDGE_MOD = 1 << 17
LEDGE_HALF = LEDGE_MOD >> 1
LEDGE_LSB_NS = 0.78125
WINDOW_LO_NS = 300.0
WINDOW_HI_NS = 500.0


def rollover_diff_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a.astype(np.int64) - b.astype(np.int64)
    d = np.where(d > LEDGE_HALF, d - LEDGE_MOD, d)
    d = np.where(d < -LEDGE_HALF, d + LEDGE_MOD, d)
    return d


def load_words(dat_path: Path) -> np.ndarray:
    raw = dat_path.read_bytes()
    whole = len(raw) - (len(raw) % WORD_SIZE)
    if whole == 0:
        return np.empty(0, dtype=np.uint64)

    words = np.frombuffer(raw[:whole], dtype=np.uint8).reshape(-1, WORD_SIZE)
    return (
        (words[:, 0].astype(np.uint64) << 32)
        | (words[:, 1].astype(np.uint64) << 24)
        | (words[:, 2].astype(np.uint64) << 16)
        | (words[:, 3].astype(np.uint64) << 8)
        | words[:, 4].astype(np.uint64)
    )


def collect_deltas(words: np.ndarray) -> tuple[np.ndarray, int]:
    if words.size == 0:
        return np.empty(0, dtype=np.int64), 0

    pos = np.arange(words.size, dtype=np.int64)
    top4 = (words >> 36) & 0xF
    low32 = words & 0xFFFFFFFF

    is_hdr = top4 == TOP4_HDR
    is_trl = top4 == TOP4_TRL
    is_trg = (top4 == TOP4_TRG) & (((words >> 30) & LOW6_MASK) == 0)
    is_err = low32 == ERR_MAGIC
    is_ovf = ((words >> 24) & 0xFF) == OVF_MAGIC
    is_hit = ~(is_hdr | is_trl | is_trg | is_err | is_ovf)

    header_pos = pos[is_hdr]
    if header_pos.size == 0:
        return np.empty(0, dtype=np.int64), 0

    trigger_pos = pos[is_trg]
    trigger_ledge = (words[is_trg] & 0x1FFFF).astype(np.int64)

    hit_pos = pos[is_hit]
    hit_ledge = ((words[is_hit] >> 8) & 0x1FFFF).astype(np.int64)

    if trigger_pos.size == 0 or hit_pos.size == 0:
        return np.empty(0, dtype=np.int64), int(header_pos.size)

    next_header = np.empty_like(header_pos)
    next_header[:-1] = header_pos[1:]
    next_header[-1] = words.size

    next_trg_idx = np.searchsorted(trigger_pos, header_pos, side="right")
    has_next_trg = next_trg_idx < trigger_pos.size
    valid_next_trg = np.zeros_like(has_next_trg, dtype=bool)
    valid_next_trg[has_next_trg] = (
        trigger_pos[next_trg_idx[has_next_trg]] < next_header[has_next_trg]
    )

    next_hit_idx = np.searchsorted(hit_pos, header_pos, side="right")
    has_next_hit = next_hit_idx < hit_pos.size
    valid_next_hit = np.zeros_like(has_next_hit, dtype=bool)
    valid_next_hit[has_next_hit] = (
        hit_pos[next_hit_idx[has_next_hit]] < next_header[has_next_hit]
    )

    keep = valid_next_trg & valid_next_hit
    if not np.any(keep):
        return np.empty(0, dtype=np.int64), int(header_pos.size)

    trg = trigger_ledge[next_trg_idx[keep]]
    hit = hit_ledge[next_hit_idx[keep]]
    return rollover_diff_vec(hit, trg), int(header_pos.size)


def main():
    print(f"reading {DAT_PATH}")
    words = load_words(DAT_PATH)
    print(f"loaded {words.size} words")

    deltas, header_count = collect_deltas(words)
    if deltas.size == 0:
        raise SystemExit("No trigger/hit pairs found with the requested header rule.")

    deltas_ns = deltas.astype(np.float64) * LEDGE_LSB_NS
    in_window = (deltas_ns >= WINDOW_LO_NS) & (deltas_ns <= WINDOW_HI_NS)
    window_entries = int(np.count_nonzero(in_window))
    window_over_headers = (window_entries / header_count) if header_count > 0 else 0.0

    print(f"headers={header_count}")
    print(f"entries_in_[{int(WINDOW_LO_NS)},{int(WINDOW_HI_NS)}]ns={window_entries}")
    print(f"entries_over_headers={window_over_headers:.6f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        deltas_ns,
        bins=np.arange(200.0, 551.0, 1.0),
        histtype="stepfilled",
        color="#2c7fb8",
        alpha=0.85,
    )
    ax.set_title("Hit Minus Trigger")
    ax.set_xlabel("hit_ledge - trigger_ledge (ns)")
    ax.set_ylabel("Count")
    ax.set_xlim(200.0, 550.0)
    ax.grid(True, alpha=0.25)

    text = (
        f"entries: {deltas.size}\n"
        f"[{int(WINDOW_LO_NS)},{int(WINDOW_HI_NS)}] ns: {window_entries}\n"
        f"/ headers ({header_count}): {window_over_headers:.6f}\n"
        f"mean: {deltas_ns.mean():.3f} ns\n"
        f"std: {deltas_ns.std():.3f} ns"
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
