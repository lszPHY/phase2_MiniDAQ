from __future__ import annotations

import os
import time
import queue
from typing import Optional

from PyQt5 import QtCore


class DatReplayThread(QtCore.QThread):
    """
    Replay a local .dat stream into the decode analysis queue.

    Expected .dat content:
      - contiguous 5-byte words (same format CaptureThread writes)
    """

    message = QtCore.pyqtSignal(str)
    stats = QtCore.pyqtSignal(
        int,  # chunk count (acts like packet counter in GUI)
        int,  # lost total (always 0 for file replay)
        int,  # total bytes pushed to analysis
        str,  # basename(dat_path)
    )
    replay_done = QtCore.pyqtSignal()

    WORD_SIZE = 5

    def __init__(
        self,
        dat_path: str,
        analysis_q: "queue.Queue[bytes]",
        chunk_bytes: int = 1024 * 256,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self.dat_path = str(dat_path)
        self.analysis_q = analysis_q
        self.chunk_bytes = max(self.WORD_SIZE, int(chunk_bytes))
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        path = self.dat_path
        if not os.path.isfile(path):
            self.message.emit(f"[Replay] File not found: {path}\n")
            self.replay_done.emit()
            return

        base = os.path.basename(path)
        self.message.emit(f"[Replay] Processing dat file: {path}\n")

        bytes_pushed = 0
        chunks = 0
        last_emit = time.time()
        carry = b""

        try:
            with open(path, "rb") as f:
                while not self._stop:
                    raw = f.read(self.chunk_bytes)
                    if not raw:
                        break

                    chunks += 1
                    buf = carry + raw
                    n_words = len(buf) // self.WORD_SIZE
                    n_take = n_words * self.WORD_SIZE

                    payload = buf[:n_take]
                    carry = buf[n_take:]

                    if payload:
                        while not self._stop:
                            try:
                                self.analysis_q.put(payload, timeout=0.1)
                                bytes_pushed += len(payload)
                                break
                            except queue.Full:
                                continue

                    now = time.time()
                    if now - last_emit >= 0.2:
                        self.stats.emit(chunks, 0, bytes_pushed, base)
                        last_emit = now

            if carry:
                self.message.emit(
                    f"[Replay] Ignored trailing {len(carry)} byte(s) not aligned to 5-byte words.\n"
                )

        except Exception as e:
            self.message.emit(f"[Replay] Error while reading file: {repr(e)}\n")
        finally:
            self.stats.emit(chunks, 0, bytes_pushed, base)
            if self._stop:
                self.message.emit("[Replay] Stopped by user.\n")
            else:
                self.message.emit("[Replay] File read complete; waiting for decode to finish.\n")
            self.replay_done.emit()
