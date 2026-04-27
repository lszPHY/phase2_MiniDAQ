# CaptureThread.py
from __future__ import annotations

import os
import time
import queue
from typing import Optional

from PyQt5 import QtCore

from pcap_session import PCapSessionHandlerPy


class CaptureThread(QtCore.QThread):
    """
    CaptureThread responsibilities:
      1) Capture raw packets from NIC
      2) Write ALL raw bytes to .dat file (ground truth)
      3) Push immutable bytes into analysis queue (non-blocking)

    This thread does:
      - NO decoding
      - NO event logic
      - NO histogramming
    """

    message = QtCore.pyqtSignal(str)
    stats = QtCore.pyqtSignal(
        int,  # total_packets
        int,  # lost_packets
        int,  # total_bytes
        str,  # output filename
    )

    def __init__(
        self,
        dev: str,
        bpf: str,
        out_path: str,
        analysis_q: "queue.Queue[bytes]",
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)

        self.dev = dev
        self.bpf = bpf
        self.out_path = out_path
        self.analysis_q = analysis_q

        self._stop = False
        self._cap: Optional[PCapSessionHandlerPy] = None
        self._fh = None

    # ------------------------------------------------------------------

    def stop(self):
        """Request thread stop (non-blocking)."""
        self._stop = True

    # ------------------------------------------------------------------

    def _open_output(self):
        out_dir = os.path.dirname(os.path.abspath(self.out_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        self._fh = open(self.out_path, "wb")
        self.message.emit(f"Writing raw data to: {self.out_path}\n")

    # ------------------------------------------------------------------

    def run(self):
        # ---------- initialize pcap ----------
        try:
            self._cap = PCapSessionHandlerPy(self.dev, self.bpf)
            self.message.emit(f"pcap initialized on {self.dev}\n")
        except Exception as e:
            self.message.emit(
                "[ERROR] Failed to initialize pcap:\n"
                f"  {repr(e)}\n\n"
                "Hint:\n"
                "  sudo setcap cap_net_raw,cap_net_admin=eip $(which python3)\n"
            )
            return

        # ---------- open output file ----------
        try:
            self._open_output()
        except Exception as e:
            self.message.emit(f"[ERROR] Failed to open output file: {repr(e)}\n")
            try:
                self._cap.close()
            except Exception:
                pass
            return

        last_stat_emit = time.time()

        # ---------- main capture loop ----------
        while not self._stop:
            try:
                data = self._cap.bufferPackets(timeout_sec=0.1)
            except Exception as e:
                self.message.emit(f"[ERROR] pcap capture failed: {repr(e)}\n")
                break

            if data.packetBuffer:
                # 1) WRITE RAW BYTES TO DISK (always)
                try:
                    self._fh.write(data.packetBuffer)
                except Exception as e:
                    self.message.emit(f"[ERROR] Write failed: {repr(e)}\n")
                    break

                # 2) PUSH IMMUTABLE COPY TO ANALYSIS (non-blocking)
                try:
                    self.analysis_q.put_nowait(bytes(data.packetBuffer))
                except queue.Full:
                    # Analysis is slower than capture ? drop decode data only
                    # Disk data is still intact
                    pass

            # ---------- emit stats periodically ----------
            now = time.time()
            if now - last_stat_emit >= 0.2:
                self.stats.emit(
                    self._cap.totalPackets,
                    self._cap.data.lostPackets,
                    self._cap.totalBufferedBytes,
                    os.path.basename(self.out_path),
                )
                last_stat_emit = now

        # ---------- cleanup ----------
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass
        self._fh = None

        try:
            if self._cap:
                self._cap.close()
        except Exception:
            pass
        self._cap = None

        self.message.emit("Capture stopped\n")
