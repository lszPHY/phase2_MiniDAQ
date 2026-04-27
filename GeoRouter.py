from typing import Dict, Tuple
from Signal import GeometryLike

class GeoRouter(GeometryLike):
    def __init__(self, by_tdcid: Dict[int, GeometryLike]):
        self.by_tdcid = dict(by_tdcid)

        # precompute tdcid -> chamber_id for O(1) lookup
        self._cid_by_tdcid: Dict[int, int] = {}
        for t, g in self.by_tdcid.items():
            self._cid_by_tdcid[int(t)] = int(getattr(g, "chamber_id", -1))

    def chamber_id_from_tdcid(self, tdc_id: int) -> int:
        return int(self._cid_by_tdcid.get(int(tdc_id), -1))

    def wire_center_from_hit(self, tdc_id: int, channel_id: int) -> Tuple[float, float, int, int]:
        g = self.by_tdcid.get(int(tdc_id))
        if g is None:
            return -1.0, -1.0, -1, -1
        return g.wire_center_from_hit(int(tdc_id), int(channel_id))