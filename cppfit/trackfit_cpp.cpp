// cppfit/trackfit_cpp.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

static inline double clamp(double x, double lo, double hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

static inline double radius_error_mm(double radius_mm) {
    constexpr double c0 = 0.24002;
    constexpr double c1 = -0.055802;
    constexpr double c2 = 0.0069947;
    constexpr double c3 = -0.000409;
    constexpr double c4 = 0.000009309;
    double sigma = c0
                 + c1 * radius_mm
                 + c2 * radius_mm * radius_mm
                 + c3 * radius_mm * radius_mm * radius_mm
                 + c4 * radius_mm * radius_mm * radius_mm * radius_mm;
    if (!std::isfinite(sigma) || sigma < 0.03) sigma = 0.03;
    return sigma;
}

static inline double weighted_mean_value(const std::vector<double>& values,
                                         const std::vector<double>& weights,
                                         double fallback) {
    if (values.empty() || values.size() != weights.size()) return fallback;
    double sum_w = 0.0;
    double sum_wx = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double v = values[i];
        const double w = weights[i];
        if (!std::isfinite(v) || !std::isfinite(w) || w <= 0.0) continue;
        sum_w += w;
        sum_wx += w * v;
    }
    if (sum_w <= 0.0) return fallback;
    return sum_wx / sum_w;
}

static inline double theta_from_slope(double slope) {
    if (!std::isfinite(slope)) return std::numeric_limits<double>::quiet_NaN();
    return -std::atan(1.0 / slope);
}

static inline bool theta_in_window(double theta, double theta_min, double theta_max) {
    if (!std::isfinite(theta)) return false;
    if (!(theta_min < theta_max)) return true;
    return theta >= theta_min && theta <= theta_max;
}

static inline bool solve_linear_3x3(const std::array<std::array<double, 3>, 3>& a_in,
                                    const std::array<double, 3>& b_in,
                                    std::array<double, 3>& x_out) {
    double aug[3][4] = {
        {a_in[0][0], a_in[0][1], a_in[0][2], b_in[0]},
        {a_in[1][0], a_in[1][1], a_in[1][2], b_in[1]},
        {a_in[2][0], a_in[2][1], a_in[2][2], b_in[2]},
    };

    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        double pivot_abs = std::abs(aug[col][col]);
        for (int row = col + 1; row < 3; ++row) {
            const double cand_abs = std::abs(aug[row][col]);
            if (cand_abs > pivot_abs) {
                pivot = row;
                pivot_abs = cand_abs;
            }
        }
        if (pivot_abs < 1.0e-16) return false;
        if (pivot != col) {
            for (int j = col; j < 4; ++j) std::swap(aug[col][j], aug[pivot][j]);
        }

        const double pivot_val = aug[col][col];
        for (int j = col; j < 4; ++j) aug[col][j] /= pivot_val;

        for (int row = 0; row < 3; ++row) {
            if (row == col) continue;
            const double factor = aug[row][col];
            if (std::abs(factor) < 1.0e-18) continue;
            for (int j = col; j < 4; ++j) aug[row][j] -= factor * aug[col][j];
        }
    }

    x_out[0] = aug[0][3];
    x_out[1] = aug[1][3];
    x_out[2] = aug[2][3];
    return std::isfinite(x_out[0]) && std::isfinite(x_out[1]) && std::isfinite(x_out[2]);
}

}  // namespace

struct LinearFitResult {
    bool ok = false;
    double slope = 0.0;
    double y_intercept = 0.0;
    double chi2 = std::numeric_limits<double>::infinity();
};

struct SeedResult {
    bool ok = false;
    double theta = 0.0;
    double slope = 0.0;
    double x_intercept = 0.0;
    double t0 = 0.0;
    double score = std::numeric_limits<double>::infinity();
    unsigned long long lr_bitmap = 0ULL;
    int lr_iters = 0;
    std::string failure_reason;
    std::string failure_detail;
};

struct FitResult {
    bool ok = false;
    bool converged = false;
    double theta = 0.0;
    double nx = 1.0;
    double ny = 0.0;
    double c = 0.0;
    double intercept_x_mm = 0.0;
    double slope = std::numeric_limits<double>::infinity();
    double y_intercept = std::numeric_limits<double>::quiet_NaN();
    double t0 = 0.0;
    double cost = std::numeric_limits<double>::infinity();
    double chi2 = std::numeric_limits<double>::infinity();
    double chi2_ndf = std::numeric_limits<double>::infinity();
    int dof = -1;
    int lr_iters = 0;
    int optimizer_iters = 0;
    double max_change = 0.0;
    double mean_abs_pull = 0.0;
    double seed_theta = 0.0;
    double seed_intercept_x_mm = 0.0;
    double seed_t0_ns = 0.0;
    unsigned long long lr_bitmap = 0ULL;
    std::string algorithm = "triggerless_exhaustive_lr";
    std::string fallback_reason;
    std::string fallback_detail;
    std::vector<int> used_idx;
};

class TrackFitter {
public:
    TrackFitter(py::array_t<double, py::array::c_style | py::array::forcecast> r_lut,
                double T0,
                double Tmax)
        : T0_(T0), Tmax_(Tmax) {
        if (r_lut.ndim() != 1) {
            throw std::runtime_error("r_lut must be 1D.");
        }
        const auto n = static_cast<int>(r_lut.shape(0));
        if (n < 2) {
            throw std::runtime_error("r_lut must have at least 2 points.");
        }
        r_.assign(r_lut.data(), r_lut.data() + n);
        n_ = n;
        dt_ = (Tmax_ - T0_) / static_cast<double>(n_ - 1);
        if (!(dt_ > 0.0)) {
            throw std::runtime_error("Invalid (Tmax - T0) for LUT.");
        }
        inv_dt_ = 1.0 / dt_;
        r_max_ = 0.0;
        for (double v : r_) {
            if (v > r_max_) r_max_ = v;
        }
    }

    double r_of_t(double t) const {
        const double tc = clamp(t, T0_, Tmax_);
        const double u = (tc - T0_) * inv_dt_;
        int i = static_cast<int>(u);
        if (i <= 0) return r_[0];
        if (i >= n_ - 1) return r_[n_ - 1];
        const double frac = u - static_cast<double>(i);
        return r_[i] + frac * (r_[i + 1] - r_[i]);
    }

    py::dict fit(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                 py::array_t<double, py::array::c_style | py::array::forcecast> y,
                 py::array_t<double, py::array::c_style | py::array::forcecast> t_corr_ns,
                 double theta_min = -1.2,
                 double theta_max = 1.2,
                 int theta_steps = 241,
                 double t0_min = -80.0,
                 double t0_max = 80.0,
                 int lr_iters = 5,
                 int t0_golden_iters = 32,
                 double max_residual_sigma = 5.0,
                 double optimizer_tolerance = 1.0e-3,
                 int optimizer_max_iters = 64) const {
        if (x.ndim() != 1 || y.ndim() != 1 || t_corr_ns.ndim() != 1) {
            throw std::runtime_error("x, y, t_corr_ns must be 1D arrays.");
        }
        const ssize_t n = x.shape(0);
        if (y.shape(0) != n || t_corr_ns.shape(0) != n) {
            throw std::runtime_error("x, y, t_corr_ns must have the same length.");
        }
        if (n < 3) {
            throw std::runtime_error("Need at least 3 hits to fit.");
        }

        const double* xp = x.data();
        const double* yp = y.data();
        const double* tp = t_corr_ns.data();

        std::vector<int> used_idx(static_cast<std::size_t>(n));
        for (int i = 0; i < static_cast<int>(n); ++i) used_idx[static_cast<std::size_t>(i)] = i;

        FitResult best = fit_triggerless(
            xp,
            yp,
            tp,
            used_idx,
            theta_min,
            theta_max,
            theta_steps,
            t0_min,
            t0_max,
            lr_iters,
            t0_golden_iters,
            max_residual_sigma,
            optimizer_tolerance,
            optimizer_max_iters
        );

        py::dict out;
        out["theta"] = best.theta;
        out["nx"] = best.nx;
        out["ny"] = best.ny;
        out["c"] = best.c;
        out["t0_ns"] = best.t0;
        out["cost"] = best.cost;
        out["lr_iters"] = best.lr_iters;
        out["optimizer_iters"] = best.optimizer_iters;
        out["max_change"] = best.max_change;
        out["mean_abs_pull"] = best.mean_abs_pull;
        out["chi2"] = best.chi2;
        out["dof"] = best.dof;
        out["chi2_ndf"] = best.chi2_ndf;
        out["converged"] = best.converged;
        out["algorithm"] = best.algorithm;
        out["fallback_reason"] = best.fallback_reason;
        out["fallback_detail"] = best.fallback_detail;
        out["intercept_x_mm"] = best.intercept_x_mm;
        out["slope"] = best.slope;
        out["y_intercept"] = best.y_intercept;
        out["seed_theta"] = best.seed_theta;
        out["seed_intercept_x_mm"] = best.seed_intercept_x_mm;
        out["seed_t0_ns"] = best.seed_t0_ns;
        out["seed_lr_bitmap"] = static_cast<double>(best.lr_bitmap);
        out["r_max_mm"] = r_max_;
        out["T0"] = T0_;
        out["Tmax"] = Tmax_;
        out["n_input"] = static_cast<int>(n);
        out["n_used"] = static_cast<int>(best.used_idx.size());
        out["used_idx"] = best.used_idx;
        return out;
    }

private:
    double radius_at_shift(double t_corr_ns, double t0_shift_ns) const {
        return r_of_t(t_corr_ns - t0_shift_ns);
    }

    LinearFitResult least_squares(const std::vector<double>& x,
                                  const std::vector<double>& y,
                                  const std::vector<double>& r) const {
        LinearFitResult out;
        const std::size_t n = x.size();
        if (n < 2 || y.size() != n || r.size() != n) {
            return out;
        }

        double x_mean = 0.0;
        double y_mean = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            x_mean += x[i];
            y_mean += y[i];
        }
        x_mean /= static_cast<double>(n);
        y_mean /= static_cast<double>(n);

        double cov = 0.0;
        double var = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dx = x[i] - x_mean;
            cov += dx * (y[i] - y_mean);
            var += dx * dx;
        }
        if (!std::isfinite(var) || std::abs(var) < 1.0e-16) return out;

        out.slope = cov / var;
        out.y_intercept = y_mean - out.slope * x_mean;
        if (!std::isfinite(out.slope) || !std::isfinite(out.y_intercept)) return out;

        const double denom = std::sqrt(out.slope * out.slope + 1.0);
        if (!(denom > 0.0) || !std::isfinite(denom)) return out;

        double chi2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dist = std::abs(x[i] * out.slope + out.y_intercept - y[i]) / denom;
            const double sigma = radius_error_mm(clamp(r[i], 0.0, r_max_));
            const double pull = dist / sigma;
            chi2 += pull * pull;
        }
        out.chi2 = chi2;
        out.ok = std::isfinite(out.chi2);
        return out;
    }

    bool gauss_newton_t0_update(const double* x,
                                const double* y,
                                const double* t,
                                const std::vector<int>& idx,
                                unsigned long long bitmap,
                                double slope,
                                double x_intercept,
                                double& t0_shift,
                                double t0_min,
                                double t0_max,
                                double max_step_ns) const {
        if (idx.empty()) return false;
        constexpr double deriv_step = 0.5;
        const double denom = std::sqrt(slope * slope + 1.0);
        if (!(denom > 0.0) || !std::isfinite(denom)) return false;

        double jt_e = 0.0;
        double jt_j = 0.0;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const int gi = idx[i];
            const double radius = radius_at_shift(t[gi], t0_shift);
            const double dist = std::abs(slope * (x[gi] - x_intercept) - y[gi]) / denom;
            const double resid = dist - radius;
            const double rp = radius_at_shift(t[gi], t0_shift + deriv_step);
            const double rm = radius_at_shift(t[gi], t0_shift - deriv_step);
            const double dr_dshift = (rp - rm) / (2.0 * deriv_step);
            const double dres_dt0 = -dr_dshift;
            jt_e += dres_dt0 * resid;
            jt_j += dres_dt0 * dres_dt0;
        }
        if (jt_j <= 1.0e-14) return false;
        double step = -jt_e / jt_j;
        step = clamp(step, -max_step_ns, max_step_ns);
        t0_shift = clamp(t0_shift + step, t0_min, t0_max);
        return std::isfinite(t0_shift);
    }

    bool weighted_intercept_update(const double* x,
                                   const double* y,
                                   const double* t,
                                   const std::vector<int>& idx,
                                   unsigned long long bitmap,
                                   double slope,
                                   double t0_shift,
                                   double& x_intercept) const {
        if (std::abs(slope) <= 1.0e-12) return false;

        std::vector<double> values;
        std::vector<double> weights;
        values.reserve(idx.size());
        weights.reserve(idx.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const int gi = idx[i];
            const bool is_right = (bitmap & (1ULL << i)) != 0ULL;
            const double sign = is_right ? 1.0 : -1.0;
            const double radius = radius_at_shift(t[gi], t0_shift);
            const double sigma = radius_error_mm(radius);
            const double w = 1.0 / (sigma * sigma);
            const double b_candidate = x[gi] + sign * radius - y[gi] / slope;
            values.push_back(b_candidate);
            weights.push_back(w);
        }
        x_intercept = weighted_mean_value(values, weights, x_intercept);
        return std::isfinite(x_intercept);
    }

    SeedResult build_triggerless_seed(const double* x,
                                      const double* y,
                                      const double* t,
                                      const std::vector<int>& idx,
                                      double theta_min,
                                      double theta_max,
                                      double t0_min,
                                      double t0_max) const {
        SeedResult best;
        const int n = static_cast<int>(idx.size());
        if (n < 3) {
            best.failure_reason = "seed_not_enough_hits";
            return best;
        }
        if (n >= 63) {
            best.failure_reason = "seed_too_many_hits";
            return best;
        }

        int ls_ok_count = 0;
        int theta0_ok_count = 0;
        int refined_ok_count = 0;
        int theta_refined_ok_count = 0;
        int score_update_count = 0;
        int best_update_count = 0;

        std::vector<double> base_r(static_cast<std::size_t>(n));
        std::vector<double> x_trial(static_cast<std::size_t>(n));
        std::vector<double> x_refined(static_cast<std::size_t>(n));
        std::vector<double> r_refined(static_cast<std::size_t>(n));

        const double seed_t0 = clamp(0.0, t0_min, t0_max);
        for (int i = 0; i < n; ++i) {
            base_r[static_cast<std::size_t>(i)] = radius_at_shift(t[idx[static_cast<std::size_t>(i)]], seed_t0);
        }

        const unsigned long long n_bitmap = 1ULL << n;
        for (unsigned long long bitmap = 0ULL; bitmap < n_bitmap; ++bitmap) {
            for (int i = 0; i < n; ++i) {
                const bool is_right = (bitmap & (1ULL << i)) != 0ULL;
                x_trial[static_cast<std::size_t>(i)] =
                    x[idx[static_cast<std::size_t>(i)]]
                    + (is_right ? base_r[static_cast<std::size_t>(i)] : -base_r[static_cast<std::size_t>(i)]);
            }

            std::vector<double> y_local(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) y_local[static_cast<std::size_t>(i)] = y[idx[static_cast<std::size_t>(i)]];
            LinearFitResult ls = least_squares(x_trial, y_local, base_r);
            if (!ls.ok || std::abs(ls.slope) < 1.0e-17) continue;
            ls_ok_count++;

            const double theta0 = theta_from_slope(ls.slope);
            if (!theta_in_window(theta0, theta_min, theta_max)) continue;
            theta0_ok_count++;

            double candidate_slope = ls.slope;
            double candidate_yint = ls.y_intercept;
            double candidate_intercept = -candidate_yint / candidate_slope;
            double candidate_t0 = seed_t0;

            double best_slope = candidate_slope;
            double best_intercept = candidate_intercept;
            double best_t0 = candidate_t0;
            double best_score = std::numeric_limits<double>::infinity();

            for (int pass = 0; pass < 2; ++pass) {
                gauss_newton_t0_update(
                    x, y, t, idx, bitmap,
                    candidate_slope, candidate_intercept,
                    candidate_t0, t0_min, t0_max, 20.0
                );

                weighted_intercept_update(
                    x, y, t, idx, bitmap,
                    candidate_slope, candidate_t0,
                    candidate_intercept
                );
                candidate_yint = -candidate_slope * candidate_intercept;

                for (int i = 0; i < n; ++i) {
                    const int gi = idx[static_cast<std::size_t>(i)];
                    const bool is_right = (bitmap & (1ULL << i)) != 0ULL;
                    const double sign = is_right ? 1.0 : -1.0;
                    const double radius = radius_at_shift(t[gi], candidate_t0);
                    x_refined[static_cast<std::size_t>(i)] = x[gi] + sign * radius;
                    r_refined[static_cast<std::size_t>(i)] = radius;
                }

                LinearFitResult refined = least_squares(x_refined, y_local, r_refined);
                if (!refined.ok || std::abs(refined.slope) < 1.0e-17) continue;
                refined_ok_count++;

                const double theta_refined = theta_from_slope(refined.slope);
                if (!theta_in_window(theta_refined, theta_min, theta_max)) continue;
                theta_refined_ok_count++;

                candidate_slope = refined.slope;
                candidate_yint = refined.y_intercept;
                candidate_intercept = -candidate_yint / candidate_slope;

                if (refined.chi2 < best_score) {
                    score_update_count++;
                    best_score = refined.chi2;
                    best_slope = candidate_slope;
                    best_intercept = candidate_intercept;
                    best_t0 = candidate_t0;
                }
            }

            if (best_score < best.score) {
                best_update_count++;
                best.ok = true;
                best.score = best_score;
                best.slope = best_slope;
                best.x_intercept = best_intercept;
                best.theta = theta_from_slope(best_slope);
                best.t0 = best_t0;
                best.lr_bitmap = bitmap;
                best.lr_iters = 2;
            }
        }

        if (!best.ok) {
            if (ls_ok_count == 0) {
                best.failure_reason = "seed_initial_lsq_failed";
            } else if (theta0_ok_count == 0) {
                best.failure_reason = "seed_initial_theta_out_of_window";
            } else if (refined_ok_count == 0) {
                best.failure_reason = "seed_refined_lsq_failed";
            } else if (theta_refined_ok_count == 0) {
                best.failure_reason = "seed_refined_theta_out_of_window";
            } else {
                best.failure_reason = "seed_no_candidate";
            }
            best.failure_detail =
                "bitmaps=" + std::to_string(n_bitmap)
                + ",ls_ok=" + std::to_string(ls_ok_count)
                + ",theta0_ok=" + std::to_string(theta0_ok_count)
                + ",refined_ok=" + std::to_string(refined_ok_count)
                + ",theta_refined_ok=" + std::to_string(theta_refined_ok_count)
                + ",score_update=" + std::to_string(score_update_count)
                + ",best_update=" + std::to_string(best_update_count);
            return best;
        }

        double prefit_slope = best.slope;
        double prefit_intercept = best.x_intercept;
        double prefit_t0 = best.t0;
        double prefit_yint = -prefit_slope * prefit_intercept;

        std::vector<double> y_local(static_cast<std::size_t>(n));
        std::vector<double> x_ref(static_cast<std::size_t>(n));
        std::vector<double> r_ref(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) y_local[static_cast<std::size_t>(i)] = y[idx[static_cast<std::size_t>(i)]];

        for (int prefit_step = 0; prefit_step < 3; ++prefit_step) {
            gauss_newton_t0_update(
                x, y, t, idx, best.lr_bitmap,
                prefit_slope, prefit_intercept,
                prefit_t0, t0_min, t0_max, 10.0
            );

            weighted_intercept_update(
                x, y, t, idx, best.lr_bitmap,
                prefit_slope, prefit_t0,
                prefit_intercept
            );
            prefit_yint = -prefit_slope * prefit_intercept;

            for (int i = 0; i < n; ++i) {
                const int gi = idx[static_cast<std::size_t>(i)];
                const bool is_right = (best.lr_bitmap & (1ULL << i)) != 0ULL;
                const double sign = is_right ? 1.0 : -1.0;
                const double radius = radius_at_shift(t[gi], prefit_t0);
                x_ref[static_cast<std::size_t>(i)] = x[gi] + sign * radius;
                r_ref[static_cast<std::size_t>(i)] = radius;
            }

            LinearFitResult refined = least_squares(x_ref, y_local, r_ref);
            if (!refined.ok || std::abs(refined.slope) < 1.0e-17) continue;
            const double theta_refined = theta_from_slope(refined.slope);
            if (!theta_in_window(theta_refined, theta_min, theta_max)) continue;

            prefit_slope = refined.slope;
            prefit_yint = refined.y_intercept;
            prefit_intercept = -prefit_yint / prefit_slope;
        }

        best.slope = prefit_slope;
        best.x_intercept = prefit_intercept;
        best.theta = theta_from_slope(prefit_slope);
        best.t0 = prefit_t0;
        best.lr_iters = 3;
        return best;
    }

    FitResult optimize_triggerless(const double* x,
                                   const double* y,
                                   const double* t,
                                   const std::vector<int>& idx,
                                   const SeedResult& seed,
                                   double theta_min,
                                   double theta_max,
                                   double t0_min,
                                   double t0_max,
                                   double max_residual_sigma,
                                   double optimizer_tolerance,
                                   int optimizer_max_iters) const {
        FitResult out;
        if (!seed.ok) {
            out.fallback_reason = "seed_invalid";
            return out;
        }

        double theta = seed.theta;
        double x_intercept = seed.x_intercept;
        double t0 = seed.t0;
        theta = clamp(theta, theta_min, theta_max);
        t0 = clamp(t0, t0_min, t0_max);

        if (optimizer_max_iters < 1) optimizer_max_iters = 1;
        if (!(optimizer_tolerance > 0.0)) optimizer_tolerance = 1.0e-3;
        if (!(max_residual_sigma > 0.0)) max_residual_sigma = 5.0;

        bool converged = false;
        double last_max_change = 0.0;
        double last_mean_abs_pull = 0.0;
        std::vector<int> last_used_idx;
        double last_chi2 = std::numeric_limits<double>::infinity();
        int last_dof = -1;
        int n_iterations = 0;

        for (int iter = 0; iter < optimizer_max_iters; ++iter) {
            const double c = std::cos(theta);
            const double s = std::sin(theta);

            std::array<std::array<double, 3>, 3> G{};
            std::array<double, 3> Y{};
            double chi2 = 0.0;
            double sum_abs_pull = 0.0;
            int n_used = 0;
            std::vector<int> used_idx_local;
            used_idx_local.reserve(idx.size());

            for (std::size_t i = 0; i < idx.size(); ++i) {
                const int gi = idx[i];
                const double signed_dist = c * (x[gi] - x_intercept) + y[gi] * s;
                const double dist = std::abs(signed_dist);
                const double radius = radius_at_shift(t[gi], t0);
                const double res = dist - radius;
                const double err = radius_error_mm(dist);
                if (!(dist > 0.0 && dist < r_max_ && std::abs(res) < max_residual_sigma * err)) {
                    continue;
                }

                const double sign = (signed_dist > 0.0) ? 1.0 : -1.0;
                const double d_theta = sign * (-s * (x[gi] - x_intercept) + y[gi] * c);
                const double d_intercept = -sign * c;
                const double d_t0 = radius_at_shift(t[gi], t0) - radius_at_shift(t[gi], t0 + 1.0);
                const double inv_var = 1.0 / (err * err);

                const std::array<double, 3> D{{d_theta, d_intercept, d_t0}};
                for (int r = 0; r < 3; ++r) {
                    for (int col = 0; col < 3; ++col) {
                        G[r][col] += D[r] * D[col] * inv_var;
                    }
                    Y[r] += res * D[r] * inv_var;
                }

                chi2 += (res * res) * inv_var;
                sum_abs_pull += std::abs(res / err);
                used_idx_local.push_back(gi);
                n_used++;
            }

            last_used_idx = used_idx_local;
            last_chi2 = chi2;
            last_dof = n_used - 3;
            last_mean_abs_pull = (n_used > 0) ? (sum_abs_pull / static_cast<double>(n_used)) : 0.0;
            n_iterations = iter + 1;

            if (n_used < 3) break;

            std::array<double, 3> rhs{{-Y[0], -Y[1], -Y[2]}};
            std::array<double, 3> delta{};
            bool solved = solve_linear_3x3(G, rhs, delta);
            if (!solved) {
                for (int k = 0; k < 3; ++k) {
                    const double diag = G[k][k];
                    if (std::abs(diag) > 1.0e-16) {
                        delta[k] = -Y[k] / diag;
                    } else {
                        delta[k] = 0.0;
                    }
                }
            }

            theta += delta[0];
            x_intercept += delta[1];
            t0 = clamp(t0 + delta[2], t0_min, t0_max);
            theta = clamp(theta, theta_min, theta_max);

            last_max_change = std::max({std::abs(delta[0]), std::abs(delta[1]), std::abs(delta[2])});
            if (last_max_change <= optimizer_tolerance) {
                converged = true;
                break;
            }
        }

        const double nx = std::cos(theta);
        const double ny = std::sin(theta);
        const double c_line = nx * x_intercept;

        out.ok = !last_used_idx.empty();
        if (!out.ok) {
            out.fallback_reason = "optimizer_no_inliers";
            out.fallback_detail =
                "optimizer_iters=" + std::to_string(n_iterations)
                + ",last_dof=" + std::to_string(last_dof)
                + ",max_residual_sigma=" + std::to_string(max_residual_sigma);
        }
        out.converged = converged;
        out.theta = theta;
        out.nx = nx;
        out.ny = ny;
        out.c = c_line;
        out.intercept_x_mm = x_intercept;
        out.t0 = t0;
        out.cost = (last_used_idx.empty()) ? std::numeric_limits<double>::infinity()
                                           : (last_chi2 / static_cast<double>(last_used_idx.size()));
        out.chi2 = last_chi2;
        out.dof = last_dof;
        out.chi2_ndf = (last_dof > 0) ? (last_chi2 / static_cast<double>(last_dof))
                                      : std::numeric_limits<double>::infinity();
        out.lr_iters = seed.lr_iters;
        out.optimizer_iters = n_iterations;
        out.max_change = last_max_change;
        out.mean_abs_pull = last_mean_abs_pull;
        out.seed_theta = seed.theta;
        out.seed_intercept_x_mm = seed.x_intercept;
        out.seed_t0_ns = seed.t0;
        out.lr_bitmap = seed.lr_bitmap;
        out.used_idx = last_used_idx;

        if (std::abs(ny) > 1.0e-12) {
            out.slope = -nx / ny;
            out.y_intercept = c_line / ny;
        } else {
            out.slope = (nx >= 0.0) ? -std::numeric_limits<double>::infinity()
                                    : std::numeric_limits<double>::infinity();
            out.y_intercept = std::numeric_limits<double>::quiet_NaN();
        }
        return out;
    }

    FitResult fit_with_theta_scan(const double* x,
                                  const double* y,
                                  const double* t,
                                  const std::vector<int>& idx,
                                  double theta_min,
                                  double theta_max,
                                  int theta_steps,
                                  double t0_min,
                                  double t0_max,
                                  int lr_iters,
                                  int t0_golden_iters,
                                  const std::string& fallback_reason,
                                  const std::string& fallback_detail = "") const {
        const int m = static_cast<int>(idx.size());
        if (m < 3) throw std::runtime_error("Need at least 3 hits to fit.");
        if (theta_steps < 2) theta_steps = 2;

        struct ScanResult {
            double theta = 0.0;
            double nx = 1.0;
            double ny = 0.0;
            double c = 0.0;
            double t0 = 0.0;
            double cost = std::numeric_limits<double>::infinity();
            int lr_iters = 0;
        };

        auto solve_for_theta_t0 = [&](double theta, double t0) {
            ScanResult out;
            out.theta = theta;
            out.nx = std::cos(theta);
            out.ny = std::sin(theta);
            out.t0 = t0;

            std::vector<double> np(static_cast<std::size_t>(m));
            std::vector<double> r(static_cast<std::size_t>(m));
            for (int i = 0; i < m; ++i) {
                const int gi = idx[static_cast<std::size_t>(i)];
                np[static_cast<std::size_t>(i)] = out.nx * x[gi] + out.ny * y[gi];
                r[static_cast<std::size_t>(i)] = radius_at_shift(t[gi], t0);
            }

            double c = 0.0;
            for (int i = 0; i < m; ++i) c += np[static_cast<std::size_t>(i)];
            c /= static_cast<double>(m);

            for (int it = 0; it < lr_iters; ++it) {
                double sum = 0.0;
                for (int i = 0; i < m; ++i) {
                    const double di = np[static_cast<std::size_t>(i)] - c;
                    const double sgn = (di >= 0.0) ? 1.0 : -1.0;
                    sum += (np[static_cast<std::size_t>(i)] - sgn * r[static_cast<std::size_t>(i)]);
                }
                c = sum / static_cast<double>(m);
                out.lr_iters = it + 1;
            }

            double cost = 0.0;
            for (int i = 0; i < m; ++i) {
                const double di = np[static_cast<std::size_t>(i)] - c;
                const double res = std::abs(di) - r[static_cast<std::size_t>(i)];
                cost += res * res;
            }
            out.c = c;
            out.cost = cost / static_cast<double>(m);
            return out;
        };

        auto best_t0_for_theta = [&](double theta) {
            const double gr = 0.6180339887498949;
            double a = t0_min;
            double b = t0_max;
            double c = b - gr * (b - a);
            double d = a + gr * (b - a);
            ScanResult fc = solve_for_theta_t0(theta, c);
            ScanResult fd = solve_for_theta_t0(theta, d);
            for (int k = 0; k < t0_golden_iters; ++k) {
                if (fc.cost <= fd.cost) {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - gr * (b - a);
                    fc = solve_for_theta_t0(theta, c);
                } else {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + gr * (b - a);
                    fd = solve_for_theta_t0(theta, d);
                }
            }
            return (fc.cost <= fd.cost) ? fc : fd;
        };

        ScanResult best_scan;
        for (int i = 0; i < theta_steps; ++i) {
            const double u = static_cast<double>(i) / static_cast<double>(theta_steps - 1);
            const double theta = theta_min + u * (theta_max - theta_min);
            ScanResult cand = best_t0_for_theta(theta);
            if (cand.cost < best_scan.cost) best_scan = cand;
        }

        FitResult out;
        out.ok = true;
        out.converged = true;
        out.theta = best_scan.theta;
        out.nx = best_scan.nx;
        out.ny = best_scan.ny;
        out.c = best_scan.c;
        out.t0 = best_scan.t0;
        out.cost = best_scan.cost;
        out.chi2 = best_scan.cost * static_cast<double>(m);
        out.dof = m - 3;
        out.chi2_ndf = (out.dof > 0) ? (out.chi2 / static_cast<double>(out.dof))
                                    : std::numeric_limits<double>::infinity();
        out.lr_iters = best_scan.lr_iters;
        out.optimizer_iters = 0;
        out.algorithm = "theta_scan_fallback";
        out.fallback_reason = fallback_reason;
        out.fallback_detail = fallback_detail;
        out.used_idx = idx;

        if (std::abs(out.nx) > 1.0e-12) {
            out.intercept_x_mm = out.c / out.nx;
        }
        if (std::abs(out.ny) > 1.0e-12) {
            out.slope = -out.nx / out.ny;
            out.y_intercept = out.c / out.ny;
        }
        return out;
    }

    FitResult fit_triggerless(const double* x,
                              const double* y,
                              const double* t,
                              const std::vector<int>& idx,
                              double theta_min,
                              double theta_max,
                              int theta_steps,
                              double t0_min,
                              double t0_max,
                              int lr_iters,
                              int t0_golden_iters,
                              double max_residual_sigma,
                              double optimizer_tolerance,
                              int optimizer_max_iters) const {
        constexpr int kMaxExhaustiveHits = 20;
        if (idx.size() > static_cast<std::size_t>(kMaxExhaustiveHits)) {
            return fit_with_theta_scan(
                x, y, t, idx,
                theta_min, theta_max, theta_steps,
                t0_min, t0_max, lr_iters, t0_golden_iters,
                "too_many_hits_for_exhaustive",
                "n_input=" + std::to_string(idx.size())
            );
        }

        SeedResult seed = build_triggerless_seed(x, y, t, idx, theta_min, theta_max, t0_min, t0_max);
        if (!seed.ok) {
            return fit_with_theta_scan(
                x, y, t, idx,
                theta_min, theta_max, theta_steps,
                t0_min, t0_max, lr_iters, t0_golden_iters,
                seed.failure_reason.empty() ? "seed_failed" : seed.failure_reason,
                seed.failure_detail
            );
        }

        FitResult fit = optimize_triggerless(
            x, y, t, idx, seed,
            theta_min, theta_max,
            t0_min, t0_max,
            max_residual_sigma,
            optimizer_tolerance,
            optimizer_max_iters
        );
        if (!fit.ok) {
            return fit_with_theta_scan(
                x, y, t, idx,
                theta_min, theta_max, theta_steps,
                t0_min, t0_max, lr_iters, t0_golden_iters,
                fit.fallback_reason.empty() ? "optimizer_failed" : fit.fallback_reason,
                fit.fallback_detail
            );
        }
        return fit;
    }

private:
    std::vector<double> r_;
    int n_ = 0;
    double T0_ = 0.0;
    double Tmax_ = 200.0;
    double dt_ = 1.0;
    double inv_dt_ = 1.0;
    double r_max_ = 0.0;
};

PYBIND11_MODULE(trackfit_cpp, m) {
    m.doc() = "Fast C++ track fitting with triggerless-style exhaustive LR seeding";

    py::class_<TrackFitter>(m, "TrackFitter")
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>, double, double>(),
             py::arg("r_lut"),
             py::arg("T0"),
             py::arg("Tmax"))
        .def("fit",
             &TrackFitter::fit,
             py::arg("x"),
             py::arg("y"),
             py::arg("t_corr_ns"),
             py::arg("theta_min") = -1.2,
             py::arg("theta_max") = 1.2,
             py::arg("theta_steps") = 241,
             py::arg("t0_min") = -80.0,
             py::arg("t0_max") = 80.0,
             py::arg("lr_iters") = 5,
             py::arg("t0_golden_iters") = 32,
             py::arg("max_residual_sigma") = 5.0,
             py::arg("optimizer_tolerance") = 1.0e-3,
             py::arg("optimizer_max_iters") = 64);
}
