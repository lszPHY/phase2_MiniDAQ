import ctypes
import ctypes.util
import select

WORD_SIZE = 5
IDLE_WORD = b"\xFF\xFF\xFF\xFF\xFF"

DATA_START = 14      # fixed (no VLAN)
POSTLOAD   = 4       # fixed trailer bytes at end of frame

PCAP_ERRBUF_SIZE = 256


class PacketData:
    __slots__ = ("packetBuffer", "lostPackets", "bufferedPackets", "lastPacket")

    def __init__(self):
        self.packetBuffer = bytearray()
        self.lostPackets = 0          # TOTAL missing packets since reset/start
        self.bufferedPackets = 0      # per-call packets (still computed, not shown)
        self.lastPacket = -1


# ---- ctypes libpcap bindings ----
_libname = ctypes.util.find_library("pcap")
if not _libname:
    raise RuntimeError("Could not find libpcap. Install libpcap (e.g. libpcap-dev).")
_pcap = ctypes.CDLL(_libname)

pcap_t_p = ctypes.c_void_p


class TimeVal(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class PcapPkthdr(ctypes.Structure):
    _fields_ = [("ts", TimeVal), ("caplen", ctypes.c_uint32), ("len", ctypes.c_uint32)]


class BpfProgram(ctypes.Structure):
    _fields_ = [("bf_len", ctypes.c_uint), ("bf_insns", ctypes.c_void_p)]


PCAP_HANDLER = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.POINTER(PcapPkthdr),
    ctypes.POINTER(ctypes.c_ubyte),
)

# prototypes
_pcap.pcap_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_pcap.pcap_create.restype = pcap_t_p

_pcap.pcap_set_snaplen.argtypes = [pcap_t_p, ctypes.c_int]
_pcap.pcap_set_snaplen.restype = ctypes.c_int

_pcap.pcap_set_promisc.argtypes = [pcap_t_p, ctypes.c_int]
_pcap.pcap_set_promisc.restype = ctypes.c_int

_pcap.pcap_set_timeout.argtypes = [pcap_t_p, ctypes.c_int]
_pcap.pcap_set_timeout.restype = ctypes.c_int

_pcap.pcap_set_buffer_size.argtypes = [pcap_t_p, ctypes.c_int]
_pcap.pcap_set_buffer_size.restype = ctypes.c_int

_pcap.pcap_activate.argtypes = [pcap_t_p]
_pcap.pcap_activate.restype = ctypes.c_int

_pcap.pcap_close.argtypes = [pcap_t_p]
_pcap.pcap_close.restype = None

_pcap.pcap_compile.argtypes = [pcap_t_p, ctypes.POINTER(BpfProgram), ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32]
_pcap.pcap_compile.restype = ctypes.c_int

_pcap.pcap_setfilter.argtypes = [pcap_t_p, ctypes.POINTER(BpfProgram)]
_pcap.pcap_setfilter.restype = ctypes.c_int

_pcap.pcap_freecode.argtypes = [ctypes.POINTER(BpfProgram)]
_pcap.pcap_freecode.restype = None

_pcap.pcap_dispatch.argtypes = [pcap_t_p, ctypes.c_int, PCAP_HANDLER, ctypes.c_void_p]
_pcap.pcap_dispatch.restype = ctypes.c_int

_pcap.pcap_geterr.argtypes = [pcap_t_p]
_pcap.pcap_geterr.restype = ctypes.c_char_p

_has_get_selectable_fd = hasattr(_pcap, "pcap_get_selectable_fd")
if _has_get_selectable_fd:
    _pcap.pcap_get_selectable_fd.argtypes = [pcap_t_p]
    _pcap.pcap_get_selectable_fd.restype = ctypes.c_int

_has_fileno = hasattr(_pcap, "pcap_fileno")
if _has_fileno:
    _pcap.pcap_fileno.argtypes = [pcap_t_p]
    _pcap.pcap_fileno.restype = ctypes.c_int


def _pcap_err(pcap_handle) -> str:
    e = _pcap.pcap_geterr(pcap_handle)
    return e.decode(errors="replace") if e else "unknown pcap error"


class PCapSessionHandlerPy:
    def __init__(
        self,
        device: str,
        bpf_filter: str,
        snaplen: int = 65536,
        promisc: int = 1,
        timeout_ms: int = 100,
        ring_buffer_bytes: int = 64 << 20,
    ):
        self.device = device
        self.bpf_filter = bpf_filter

        self.checkPackets = True
        self.data = PacketData()

        self.totalPackets = 0
        self.totalBufferedBytes = 0  # TOTAL extracted non-idle bytes

        errbuf = ctypes.create_string_buffer(PCAP_ERRBUF_SIZE)
        self._pcap = _pcap.pcap_create(device.encode(), errbuf)
        if not self._pcap:
            raise RuntimeError(f"pcap_create failed: {errbuf.value.decode(errors='replace')}")

        def _chk(ret, name):
            if ret != 0:
                raise RuntimeError(f"{name} failed (ret={ret}): {_pcap_err(self._pcap)}")

        _chk(_pcap.pcap_set_snaplen(self._pcap, int(snaplen)), "pcap_set_snaplen")
        _chk(_pcap.pcap_set_promisc(self._pcap, int(promisc)), "pcap_set_promisc")
        _chk(_pcap.pcap_set_timeout(self._pcap, int(timeout_ms)), "pcap_set_timeout")
        _chk(_pcap.pcap_set_buffer_size(self._pcap, int(ring_buffer_bytes)), "pcap_set_buffer_size")

        act = _pcap.pcap_activate(self._pcap)
        if act < 0:
            msg = _pcap_err(self._pcap)
            _pcap.pcap_close(self._pcap)
            self._pcap = None
            raise RuntimeError(f"pcap_activate failed (ret={act}): {msg}")

        prog = BpfProgram()
        netmask = ctypes.c_uint32(0xFFFFFF)
        if _pcap.pcap_compile(self._pcap, ctypes.byref(prog), bpf_filter.encode(), 1, netmask) < 0:
            msg = _pcap_err(self._pcap)
            _pcap.pcap_close(self._pcap)
            self._pcap = None
            raise RuntimeError(f"pcap_compile failed: {msg}")

        if _pcap.pcap_setfilter(self._pcap, ctypes.byref(prog)) < 0:
            msg = _pcap_err(self._pcap)
            _pcap.pcap_freecode(ctypes.byref(prog))
            _pcap.pcap_close(self._pcap)
            self._pcap = None
            raise RuntimeError(f"pcap_setfilter failed: {msg}")

        _pcap.pcap_freecode(ctypes.byref(prog))

        fd = -1
        if _has_get_selectable_fd:
            fd = _pcap.pcap_get_selectable_fd(self._pcap)
        if fd < 0 and _has_fileno:
            fd = _pcap.pcap_fileno(self._pcap)
        if fd < 0:
            raise RuntimeError("Could not obtain selectable fd from libpcap.")

        self._fd = fd
        self._cb = PCAP_HANDLER(self._ctypes_cb)  # MUST keep reference

    def close(self):
        if self._pcap:
            _pcap.pcap_close(self._pcap)
            self._pcap = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def setCheckPackets(self, val: bool):
        self.checkPackets = bool(val)

    def resetCounters(self):
        """Optional: reset all cumulative counters."""
        self.data.lastPacket = -1
        self.data.lostPackets = 0
        self.totalPackets = 0
        self.totalBufferedBytes = 0

    def _handle_one_packet(self, pkt: bytes):
        caplen = len(pkt)
        if caplen < (DATA_START + POSTLOAD + 2):
            return

        start = DATA_START
        end = caplen - POSTLOAD
        if end <= start:
            return

        mv = memoryview(pkt)
        buf = self.data.packetBuffer
        idle = IDLE_WORD

        for off in range(start, end, WORD_SIZE):
            w = mv[off:off + WORD_SIZE].tobytes()
            if w != idle:
                buf.extend(w)
                self.totalBufferedBytes += WORD_SIZE

        packetNum = (pkt[caplen - 2] << 8) | pkt[caplen - 1]

        if self.checkPackets and self.data.lastPacket != -1:
            expected = (self.data.lastPacket + 1) & 0xFFFF
            if packetNum != expected:
                missing = (packetNum - expected) & 0xFFFF
                self.data.lostPackets += missing

        self.data.lastPacket = packetNum
        self.totalPackets += 1

    def _ctypes_cb(self, user, hdr_p, pkt_p):
        try:
            hdr = hdr_p.contents
            pkt = ctypes.string_at(pkt_p, hdr.caplen)
            self._handle_one_packet(pkt)
        except Exception:
            return

    def bufferPackets(self, timeout_sec: float = 0.1) -> PacketData:
        """
        Drives capture. packetBuffer is per-call scratch (cleared here),
        but TOTALS are kept in:
          - self.totalPackets
          - self.totalBufferedBytes
          - self.data.lostPackets
        """
        self.data.packetBuffer.clear()
        self.data.bufferedPackets = 0  # not displayed, but harmless to keep

        r, _, _ = select.select([self._fd], [], [], float(timeout_sec))
        if not r:
            return self.data

        n = _pcap.pcap_dispatch(self._pcap, -1, self._cb, None)
        if n > 0:
            self.data.bufferedPackets = int(n)
        return self.data
