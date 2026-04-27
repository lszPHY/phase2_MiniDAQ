# rt_function.py
# -*- coding: utf-8 -*-
#
# ROOT-free RTFunction that loads Chebyshev coefficients from a plain txt file:
#   - txt contains only c0..c(N-1), one float per line (comments/blank ok)
#   - N is optional (if provided, will be validated)
#   - T0/Tmax come from constructor

from __future__ import annotations

import os
import numpy as np
from numpy.polynomial.chebyshev import chebval


class RTFunction:
    def __init__(
        self,
        txt_file: str,
        *,
        t_min: float = 0.0,
        t_max: float = 200.0,
        n_coeff: int = 10,  
    ):
        self.txt_file = str(txt_file)
        self.T0 = float(t_min)
        self.Tmax = float(t_max)

        self.params: np.ndarray | None = None
        self.r_lut_sorted: np.ndarray | None = None
        self.t_lut_sorted: np.ndarray | None = None

        self._load_coefficients_from_txt(n_coeff=n_coeff)

    # -------------------------------------------------------------------------

    def _load_coefficients_from_txt(self, *, n_coeff: int | None) -> None:
        if not os.path.isfile(self.txt_file):
            raise RuntimeError(f"[RT] TXT file not found: {self.txt_file}")

        coeffs: list[float] = []
        with open(self.txt_file, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                # allow inline comments: "1.23  # comment"
                s = s.split("#", 1)[0].strip()
                if not s:
                    continue
                coeffs.append(float(s))

        if not coeffs:
            raise RuntimeError(f"[RT] No coefficients found in: {self.txt_file}")

        if n_coeff is not None and len(coeffs) != int(n_coeff):
            raise RuntimeError(
                f"[RT] Coefficient count mismatch in {self.txt_file}: "
                f"expected N={int(n_coeff)}, got {len(coeffs)}"
            )

        self.params = np.asarray(coeffs, dtype=np.float64)

        print(f"[RT] Loaded N={len(self.params)} Chebyshev coefficients from {self.txt_file}")
        print(f"[RT] t-range: [{self.T0}, {self.Tmax}] ns")

    # -------------------------------------------------------------------------

    def r_of_t(self, t):
        """r(t) in mm from Chebyshev coefficients."""
        if self.params is None:
            raise RuntimeError("[RT] params not loaded")

        t = np.asarray(t, dtype=np.float64)
        x = (2.0 * t - (self.Tmax + self.T0)) / (self.Tmax - self.T0)
        return chebval(x, self.params)

    # -------------------------------------------------------------------------

    def build_fast_inverse(self, n: int = 2048):
        """Build t(r) inverse LUT for fast interpolation."""
        if self.params is None:
            raise RuntimeError("[RT] params not loaded")

        n = int(n)
        print(f"[RT] Building inverse LUT n={n} for t=[{self.T0},{self.Tmax}] ns")

        t_lut = np.linspace(self.T0, self.Tmax, n, dtype=np.float64)
        r_lut = self.r_of_t(t_lut)

        order = np.argsort(r_lut)
        self.r_lut_sorted = np.maximum(r_lut[order].astype(np.float32), 0.0)
        self.t_lut_sorted = t_lut[order].astype(np.float32)

        print(f"[RT] LUT r-range: [{self.r_lut_sorted[0]:.4f}, {self.r_lut_sorted[-1]:.4f}] mm")

    def t_from_r_fast(self, r_mm):
        """Fast inverse t(r) using LUT + np.interp."""
        if self.r_lut_sorted is None or self.t_lut_sorted is None:
            raise RuntimeError("Call build_fast_inverse() first.")

        r = np.asarray(r_mm, dtype=np.float32)
        t = np.full_like(r, np.nan, dtype=np.float32)

        rmin = float(self.r_lut_sorted[0])
        rmax = float(self.r_lut_sorted[-1])
        m = (r >= rmin) & (r <= rmax)
        if np.any(m):
            t[m] = np.interp(r[m], self.r_lut_sorted, self.t_lut_sorted).astype(np.float32)
        return t
