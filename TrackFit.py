from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import os
import sys
import numpy as np

try:
    import trackfit_cpp  # type: ignore
except Exception as exc:  # pragma: no cover
    _first_import_error: Exception = exc
    _cppfit_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cppfit")
    if os.path.isdir(_cppfit_dir) and _cppfit_dir not in sys.path:
        sys.path.insert(0, _cppfit_dir)

    try:
        import trackfit_cpp  # type: ignore
    except Exception as exc2:  # pragma: no cover
        trackfit_cpp = None
        _IMPORT_ERROR: Optional[Exception] = exc2
        _IMPORT_ERROR_DETAIL = repr(_first_import_error)
    else:
        _IMPORT_ERROR = None
        _IMPORT_ERROR_DETAIL = None
else:
    _IMPORT_ERROR = None
    _IMPORT_ERROR_DETAIL = None


@dataclass(frozen=True, slots=True)
class FitParams:
    theta_min: float = -1.2
    theta_max: float = 1.2
    theta_steps: int = 241
    # Optional hard limits on top of adaptive event-by-event t0 bounds.
    # Defaults are wide enough to effectively disable extra hard bounds.
    t0_min: float = -1.0e9
    t0_max: float = 1.0e9
    # Optional margin for strict feasible interval:
    #   t0 in [max(t_corr)-Tmax+margin, min(t_corr)-T0-margin]
    t0_bound_margin_ns: float = 0.0
    # Legacy scan knobs retained for the C++ fallback path.
    lr_iters: int = 5
    t0_golden_iters: int = 32
    # Triggerless-style iterative fit controls.
    max_residual_sigma: float = 5.0
    optimizer_tolerance: float = 1.0e-3
    optimizer_max_iters: int = 64


@dataclass(frozen=True, slots=True)
class TrackFitResult:
    chamber_id: int
    ok: bool
    reason: str
    n_input: int
    n_used: int
    values: Mapping[str, Any] = field(default_factory=dict)


class TrackFit:
    """
    Thin Python wrapper around trackfit_cpp.TrackFitter.

    Inputs:
      - hit x/y from event chamber hits
      - corrected times from EventQuality TimeWindowResult (t_corr_ns)
      - optional keep_mask from TimeWindowResult
    """

    def __init__(
        self,
        *,
        r_lut: Sequence[float],
        T0: float,
        Tmax: float,
        min_hits: int = 6,
        fit_only_kept_hits: bool = True,
        fit_params: Optional[FitParams] = None,
    ):
        if trackfit_cpp is None:  # pragma: no cover
            raise RuntimeError(
                "trackfit_cpp import failed. "
                f"Original error: {repr(_IMPORT_ERROR)}"
                + (f"; first import error: {_IMPORT_ERROR_DETAIL}" if _IMPORT_ERROR_DETAIL else "")
            )

        lut = np.asarray(r_lut, dtype=np.float64)
        if lut.ndim != 1 or lut.size < 2:
            raise ValueError("r_lut must be a 1D array with at least 2 entries.")

        self._r_lut = np.ascontiguousarray(lut, dtype=np.float64)
        self._lut_n = int(self._r_lut.size)
        self.T0 = float(T0)
        self.Tmax = float(Tmax)
        if not (self.Tmax > self.T0):
            raise ValueError(f"Invalid RT range: T0={self.T0}, Tmax={self.Tmax}")
        self._inv_dt = float(self._lut_n - 1) / (self.Tmax - self.T0)
        self.min_hits = int(min_hits)
        self.fit_only_kept_hits = bool(fit_only_kept_hits)
        self.fit_params = fit_params or FitParams()
        self._fitter = trackfit_cpp.TrackFitter(self._r_lut, self.T0, self.Tmax)

    @staticmethod
    def available() -> bool:
        return trackfit_cpp is not None

    @classmethod
    def from_rt(
        cls,
        rt: Any,
        *,
        n_lut: int = 2048,
        min_hits: int = 6,
        fit_only_kept_hits: bool = True,
        fit_params: Optional[FitParams] = None,
    ) -> "TrackFit":
        if rt is None:
            raise ValueError("rt cannot be None.")

        T0 = float(getattr(rt, "T0"))
        Tmax = float(getattr(rt, "Tmax"))
        if not (Tmax > T0):
            raise ValueError(f"Invalid RT range: T0={T0}, Tmax={Tmax}")

        n = max(64, int(n_lut))
        t = np.linspace(T0, Tmax, n, dtype=np.float64)
        r = np.asarray(rt.r_of_t(t), dtype=np.float64)
        if r.shape != t.shape:
            raise ValueError("rt.r_of_t(t) returned invalid shape.")

        # Keep fitter numerically stable if RT has tiny negatives from polynomial edges.
        r = np.clip(r, 0.0, None)

        return cls(
            r_lut=r,
            T0=T0,
            Tmax=Tmax,
            min_hits=min_hits,
            fit_only_kept_hits=fit_only_kept_hits,
            fit_params=fit_params,
        )

    @staticmethod
    def _normalize_fit_out(raw: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in dict(raw).items():
            key = str(k)

            if isinstance(v, (bool, list, tuple, dict, str)):
                out[key] = v
                continue

            if isinstance(v, np.generic):
                if np.issubdtype(v.dtype, np.bool_):
                    out[key] = bool(v)
                else:
                    out[key] = float(v)
                continue

            if isinstance(v, (int, float)):
                out[key] = float(v)
                continue

            try:
                out[key] = float(v)
            except Exception:
                out[key] = v

        return out

    def radius_from_time_ns(self, t_ns: Any):
        """
        RT lookup with the same linear interpolation and clamping as TrackFitter.
        Returns radius in mm for each input time.
        """
        t = np.asarray(t_ns, dtype=np.float64)
        scalar = (t.ndim == 0)
        if scalar:
            t = t.reshape(1)

        tc = np.clip(t, self.T0, self.Tmax)
        u = (tc - self.T0) * self._inv_dt

        i = np.floor(u).astype(np.int64)
        i = np.clip(i, 0, self._lut_n - 2)
        frac = u - i

        r0 = self._r_lut[i]
        r1 = self._r_lut[i + 1]
        out = r0 + frac * (r1 - r0)

        if scalar:
            return float(out[0])
        return out

    def _fit_arrays(self, x: np.ndarray, y: np.ndarray, t_corr_ns: np.ndarray) -> Dict[str, Any]:
        p = self.fit_params

        t = np.asarray(t_corr_ns, dtype=np.float64)
        if t.ndim != 1 or t.size == 0:
            raise ValueError("t_corr_ns must be a non-empty 1D array.")

        t_min = float(np.nanmin(t))
        t_max = float(np.nanmax(t))
        margin = max(0.0, float(p.t0_bound_margin_ns))

        # Strict feasible bounds for convention t_real = t_corr - t0:
        # ensure ALL fitted hits satisfy T0 <= t_real <= Tmax.
        t0_lo = t_max - self.Tmax + margin
        t0_hi = t_min - self.T0 - margin

        # Optional extra hard bounds if user provides tighter limits.
        hard_lo = float(p.t0_min)
        hard_hi = float(p.t0_max)
        if hard_lo < hard_hi:
            t0_lo = max(t0_lo, hard_lo)
            t0_hi = min(t0_hi, hard_hi)

        if not (np.isfinite(t0_lo) and np.isfinite(t0_hi) and (t0_lo < t0_hi)):
            raise ValueError("no_feasible_t0_range")

        out = self._fitter.fit(
            x=x,
            y=y,
            t_corr_ns=t_corr_ns,
            theta_min=float(p.theta_min),
            theta_max=float(p.theta_max),
            theta_steps=int(p.theta_steps),
            t0_min=float(t0_lo),
            t0_max=float(t0_hi),
            lr_iters=int(p.lr_iters),
            t0_golden_iters=int(p.t0_golden_iters),
            max_residual_sigma=float(p.max_residual_sigma),
            optimizer_tolerance=float(p.optimizer_tolerance),
            optimizer_max_iters=int(p.optimizer_max_iters),
        )
        out["t0_min_used"] = float(t0_lo)
        out["t0_max_used"] = float(t0_hi)
        return self._normalize_fit_out(out)

    def fit_chamber_hits(
        self,
        *,
        chamber_id: int,
        hits: Sequence[Any],
        t_corr_ns: Sequence[float],
        keep_mask: Optional[Sequence[bool]] = None,
    ) -> TrackFitResult:
        n_input = len(hits)
        if n_input == 0:
            return TrackFitResult(chamber_id, False, "no_hits", 0, 0, {})

        t_arr = np.asarray(t_corr_ns, dtype=np.float64)
        if t_arr.ndim != 1 or t_arr.shape[0] != n_input:
            return TrackFitResult(chamber_id, False, "time_size_mismatch", n_input, 0, {})

        x_arr = np.asarray([float(getattr(h, "x", np.nan)) for h in hits], dtype=np.float64)
        y_arr = np.asarray([float(getattr(h, "y", np.nan)) for h in hits], dtype=np.float64)

        good = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(t_arr)
        good &= (x_arr >= 0.0) & (y_arr >= 0.0)

        if keep_mask is not None and self.fit_only_kept_hits:
            keep = np.asarray(keep_mask, dtype=bool)
            if keep.ndim != 1 or keep.shape[0] != n_input:
                return TrackFitResult(chamber_id, False, "keep_mask_size_mismatch", n_input, 0, {})
            good &= keep

        n_used = int(np.count_nonzero(good))
        if n_used < max(3, self.min_hits):
            return TrackFitResult(chamber_id, False, "not_enough_hits", n_input, n_used, {})

        src_idx = np.flatnonzero(good).astype(np.int64)

        try:
            vals = self._fit_arrays(x_arr[good], y_arr[good], t_arr[good])
        except ValueError as exc:
            if str(exc) == "no_feasible_t0_range":
                return TrackFitResult(chamber_id, False, "no_feasible_t0_range", n_input, n_used, {})
            return TrackFitResult(chamber_id, False, f"fit_exception:{type(exc).__name__}", n_input, n_used, {})
        except Exception as exc:
            return TrackFitResult(chamber_id, False, f"fit_exception:{type(exc).__name__}", n_input, n_used, {})

        used_mask_input = np.zeros(n_input, dtype=bool)
        used_local = vals.get("used_idx", None)
        if isinstance(used_local, (list, tuple)):
            for j in used_local:
                try:
                    jl = int(j)
                except Exception:
                    continue
                if 0 <= jl < src_idx.size:
                    used_mask_input[int(src_idx[jl])] = True
        else:
            used_mask_input[good] = True

        vals["used_mask_input"] = used_mask_input.astype(np.uint8).tolist()
        vals["used_indices_input"] = np.flatnonzero(used_mask_input).astype(np.int64).tolist()

        n_used_final = int(round(vals.get("n_used", float(n_used))))
        try:
            dof = int(round(float(vals.get("dof", n_used_final - 3))))
        except Exception:
            dof = n_used_final - 3
        if dof < 0 and n_used_final >= 3:
            vals["dof"] = float(n_used_final - 3)
            chi2 = vals.get("chi2", None)
            try:
                chi2_f = float(chi2)
            except Exception:
                chi2_f = float("nan")
            vals["chi2_ndf"] = (
                float(chi2_f / (n_used_final - 3))
                if (n_used_final > 3 and np.isfinite(chi2_f))
                else float("inf")
            )
        return TrackFitResult(chamber_id, True, "ok", n_input, n_used_final, vals)

    def fit_event(
        self,
        *,
        ev: Any,
        time_by_chamber: Mapping[int, Any],
        valid_by_chamber: Optional[Mapping[int, bool]] = None,
        fit_invalid_chambers: bool = False,
    ) -> Dict[int, TrackFitResult]:
        out: Dict[int, TrackFitResult] = {}
        chambers = getattr(ev, "chambers", None)
        if not isinstance(chambers, dict):
            return out

        for cid_raw, block in chambers.items():
            cid = int(cid_raw)
            hits = getattr(block, "hits", ())
            tw = time_by_chamber.get(cid) if time_by_chamber is not None else None

            if tw is None:
                out[cid] = TrackFitResult(cid, False, "missing_timewindow", len(hits), 0, {})
                continue

            if valid_by_chamber is not None and not fit_invalid_chambers:
                if not bool(valid_by_chamber.get(cid, True)):
                    out[cid] = TrackFitResult(cid, False, "chamber_invalid", len(hits), 0, {})
                    continue

            t_corr = getattr(tw, "t_corr_ns", None)
            keep = getattr(tw, "keep_mask", None)
            if t_corr is None:
                out[cid] = TrackFitResult(cid, False, "missing_t_corr_ns", len(hits), 0, {})
                continue

            out[cid] = self.fit_chamber_hits(
                chamber_id=cid,
                hits=hits,
                t_corr_ns=t_corr,
                keep_mask=keep,
            )

        return out
