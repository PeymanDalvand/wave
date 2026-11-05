"""Shared seismic-physics building blocks for the Manim demos in this repository.

Every scene in this project animates a different facet of exploration
seismology (source wavelets, layered media, absorbing boundaries, NMO,
deconvolution, ...). Rather than re-deriving finite-difference stencils and
Wiener-Hopf systems independently in each scene file, the numerical models
live here once, are unit-testable in isolation, and are simply driven by the
animation code.

The 2D acoustic solver (:class:`AcousticWaveSolver2D`) builds its Laplacian
operator with `findiff <https://github.com/maroba/findiff>`_, a Windows/CI
friendly, pure-Python finite-difference library, instead of hand-rolled
array slicing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from findiff import Diff
from scipy.linalg import solve_toeplitz


# ---------------------------------------------------------------------------
# Source wavelets
# ---------------------------------------------------------------------------

def ricker_wavelet(t, frequency: float = 15.0, amplitude: float = 1.0):
    """Zero-phase Ricker ("Mexican hat") wavelet, the standard seismic source pulse.

    The pulse is centered at ``t = 1 / frequency`` so it starts and ends near
    zero for ``t >= 0``, which makes it suitable as a causal source signature.

    Args:
        t: Time (scalar or array), in seconds.
        frequency: Dominant/peak frequency of the wavelet, in Hz.
        amplitude: Peak scaling factor.
    """
    t0 = 1.0 / frequency
    tau = np.asarray(t) - t0
    pi_f_tau_sq = (np.pi * frequency * tau) ** 2
    return amplitude * (1.0 - 2.0 * pi_f_tau_sq) * np.exp(-pi_f_tau_sq)


# ---------------------------------------------------------------------------
# Shared visual helper: zigzag spring geometry
# ---------------------------------------------------------------------------

def spring_zigzag_points(
    start: np.ndarray, end: np.ndarray, n_coils: int = 10, width: float = 0.3
) -> list[np.ndarray]:
    """Return the polyline points of a 2D zigzag "spring" drawn from ``start`` to ``end``."""
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return [start, end]

    unit_dir = direction / length
    perp_dir = np.array([-unit_dir[1], unit_dir[0], 0.0])

    n_segments = n_coils * 2
    points = [start]
    for i in range(1, n_segments):
        t = i / n_segments
        offset = width * (1.0 if i % 2 == 0 else -1.0)
        points.append(start + direction * t + perp_dir * offset)
    points.append(end)
    return points


# ---------------------------------------------------------------------------
# 1D vertical mass-spring chain (gravity-loaded lumped-parameter medium)
# ---------------------------------------------------------------------------

@dataclass
class MassSpringChain1D:
    """A chain of ``n`` masses hanging from a fixed anchor, coupled by springs.

    This is the lumped-parameter (discrete) analogue of a 1D elastic medium
    under gravity: as ``n`` grows and per-block mass/stiffness are held to a
    fixed total, the chain's free vibrations approximate wave propagation
    through a layered rod, including reflection at velocity/impedance
    contrasts (:attr:`stiff_band`) and one-way absorption at the far end
    (:attr:`sponge_start_frac`).

    Attributes:
        n: Number of discrete masses.
        total_length: Combined rest length of all springs.
        total_mass: Combined mass of all blocks.
        base_stiffness: Reference stiffness; each spring's stiffness is
            ``base_stiffness * n`` so the chain's effective wave speed stays
            comparable across different discretizations.
        gravity: Gravitational acceleration.
        anchor_y: Fixed y-coordinate of the ceiling anchor.
        stiff_band: Optional ``(start_frac, end_frac)`` of the chain (as a
            fraction of ``n``) whose stiffness is multiplied by
            :attr:`stiff_multiplier`, modeling a higher-velocity layer.
        sponge_start_frac: Fraction of the chain (from the top) after which
            damping is quadratically ramped up to :attr:`sponge_max_damping`,
            modeling an absorbing boundary.
        base_damping: Damping coefficient applied to every mass.
    """

    n: int
    total_length: float
    total_mass: float
    base_stiffness: float
    gravity: float = 9.8
    anchor_y: float = 3.5
    stiff_band: Optional[tuple[float, float]] = None
    stiff_multiplier: float = 8.0
    sponge_start_frac: float = 1.0
    sponge_max_damping: float = 25.0
    base_damping: float = 0.02

    def __post_init__(self) -> None:
        n = self.n
        self.mass = np.full(n, self.total_mass / n)
        self.stiffness = np.full(n, self.base_stiffness * n)
        self.rest_length = np.full(n, self.total_length / n)

        if self.stiff_band is not None:
            start = int(n * self.stiff_band[0])
            end = int(n * self.stiff_band[1])
            self.stiffness[start:end] *= self.stiff_multiplier

        self.sponge_start = int(n * self.sponge_start_frac)
        sponge_pad = n - self.sponge_start
        self.damping = np.full(n, self.base_damping)
        if sponge_pad > 0:
            taper = (np.arange(sponge_pad) / sponge_pad) ** 2
            self.damping[self.sponge_start :] += taper * self.sponge_max_damping

        self.equilibrium = self._equilibrium_positions()
        self.position = self.equilibrium.copy()
        self.velocity = np.zeros(n)
        self.time = 0.0

    def _equilibrium_positions(self) -> np.ndarray:
        """Static positions where spring tension exactly balances gravity."""
        mass_below = np.cumsum(self.mass[::-1])[::-1]
        stretch = (mass_below * self.gravity) / self.stiffness
        return self.anchor_y - np.cumsum(self.rest_length + stretch)

    def apply_gaussian_pulse(self, center_index: float, width: float, amplitude: float = 0.6) -> None:
        """Displace the chain by an initial Gaussian pulse centered on ``center_index``."""
        offsets = (np.arange(self.n) - center_index) / width
        self.position -= amplitude * np.exp(-0.5 * offsets**2)

    def displacement(self) -> np.ndarray:
        """Current displacement from equilibrium, i.e. the seismic "trace" signal."""
        return self.position - self.equilibrium

    def step(self, dt: float, dt_sim: float = 0.001) -> np.ndarray:
        """Advance the chain by ``dt`` seconds using sub-stepped semi-implicit Euler.

        ``dt`` is internally divided into fixed-size sub-steps of at most
        ``dt_sim`` seconds for numerical stability, mirroring the CFL-style
        sub-stepping used throughout classical explicit FD wave codes.
        """
        steps = max(1, int(dt / dt_sim) + 1)
        actual_dt = dt / steps
        for _ in range(steps):
            self._substep(actual_dt)
        return self.position

    def advance(self, dt: float) -> np.ndarray:
        """Advance the chain by exactly one fixed-size semi-implicit Euler step.

        Intended for offline precomputation of long time series at a known,
        fixed simulation time step (as opposed to :meth:`step`, which
        sub-divides an arbitrary animation-frame ``dt``).
        """
        self._substep(dt)
        return self.position

    def _substep(self, dt: float) -> None:
        y, v = self.position, self.velocity
        stretch_from_anchor = (self.anchor_y - y[0]) - self.rest_length[0]
        stretch = y[:-1] - y[1:] - self.rest_length[1:]

        force_above = np.zeros(self.n)
        force_above[0] = self.stiffness[0] * stretch_from_anchor
        force_above[1:] = self.stiffness[1:] * stretch

        force_below = np.zeros(self.n)
        force_below[:-1] = self.stiffness[1:] * stretch

        accel = (force_above - force_below - self.mass * self.gravity - self.damping * v) / self.mass
        v += accel * dt
        y += v * dt
        self.time += dt


# ---------------------------------------------------------------------------
# 2D mass-spring grid (vertical + horizontal coupling)
# ---------------------------------------------------------------------------

@dataclass
class MassSpringGrid2D:
    """A 2D lattice of gravity-loaded masses: a coarse lumped-parameter model
    of a layered medium with lateral coupling, used to visualize shot gathers.
    """

    n_rows: int
    n_cols: int
    total_length: float
    total_mass: float
    base_stiffness_y: float
    base_stiffness_x: float
    gravity: float = 9.8
    anchor_y: float = 3.5
    stiff_band: Optional[tuple[float, float]] = None
    stiff_multiplier: float = 8.0
    sponge_start_frac: float = 0.76
    sponge_max_damping: float = 25.0
    base_damping: float = 0.05

    def __post_init__(self) -> None:
        n_rows, n_cols = self.n_rows, self.n_cols
        self.mass = np.full((n_rows, n_cols), self.total_mass / n_rows)
        self.stiffness_y = np.full((n_rows, n_cols), self.base_stiffness_y * n_rows)
        self.stiffness_x = self.base_stiffness_x * n_cols
        self.rest_length = np.full((n_rows, n_cols), self.total_length / n_rows)

        if self.stiff_band is not None:
            start = int(n_rows * self.stiff_band[0])
            end = int(n_rows * self.stiff_band[1])
            self.stiffness_y[start:end, :] *= self.stiff_multiplier

        self.sponge_start = int(n_rows * self.sponge_start_frac)
        sponge_pad = n_rows - self.sponge_start
        self.damping = np.full((n_rows, n_cols), self.base_damping)
        if sponge_pad > 0:
            taper = (np.arange(sponge_pad) / sponge_pad) ** 2
            self.damping[self.sponge_start :, :] += taper[:, None] * self.sponge_max_damping

        self.equilibrium = self._equilibrium_positions()
        self.position = self.equilibrium.copy()
        self.velocity = np.zeros((n_rows, n_cols))
        self.time = 0.0

    def _equilibrium_positions(self) -> np.ndarray:
        mass_below = np.cumsum(self.mass[::-1, :], axis=0)[::-1, :]
        stretch = (mass_below * self.gravity) / self.stiffness_y
        return self.anchor_y - np.cumsum(self.rest_length + stretch, axis=0)

    def apply_gaussian_pulse(
        self, center_row: float, center_col: float, row_width: float, col_width: float, amplitude: float = 1.0
    ) -> None:
        rows = np.arange(self.n_rows)[:, None]
        cols = np.arange(self.n_cols)[None, :]
        dist_sq = ((rows - center_row) / row_width) ** 2 + ((cols - center_col) / col_width) ** 2
        self.position -= amplitude * np.exp(-0.5 * dist_sq)

    def displacement(self) -> np.ndarray:
        return self.position - self.equilibrium

    def step(self, dt: float, dt_sim: float = 0.002) -> np.ndarray:
        steps = max(1, int(dt / dt_sim) + 1)
        actual_dt = dt / steps
        for _ in range(steps):
            self._substep(actual_dt)
        return self.position

    def _substep(self, dt: float) -> None:
        y, v = self.position, self.velocity
        stretch_from_anchor = (self.anchor_y - y[0, :]) - self.rest_length[0, :]
        stretch = y[:-1, :] - y[1:, :] - self.rest_length[1:, :]

        force_above = np.zeros_like(y)
        force_above[0, :] = self.stiffness_y[0, :] * stretch_from_anchor
        force_above[1:, :] = self.stiffness_y[1:, :] * stretch

        force_below = np.zeros_like(y)
        force_below[:-1, :] = self.stiffness_y[1:, :] * stretch

        force_left = np.zeros_like(y)
        force_left[:, 1:] = self.stiffness_x * (y[:, :-1] - y[:, 1:])

        force_right = np.zeros_like(y)
        force_right[:, :-1] = self.stiffness_x * (y[:, 1:] - y[:, :-1])

        accel = (
            force_above - force_below + force_left + force_right - self.mass * self.gravity - self.damping * v
        ) / self.mass
        v += accel * dt
        y += v * dt
        self.time += dt


# ---------------------------------------------------------------------------
# 1D driven transverse chain (no gravity) -- for impedance-boundary studies
# ---------------------------------------------------------------------------

@dataclass
class DrivenTransverseChain1D:
    """A horizontal chain of masses with a prescribed (source-driven) left end
    and a free right end, used to visualize reflection/transmission at an
    impedance (mass/density) contrast.
    """

    mass: np.ndarray
    stiffness: float
    damping: float = 0.05

    def __post_init__(self) -> None:
        n = len(self.mass)
        self.position = np.zeros(n)
        self.velocity = np.zeros(n)
        self.time = 0.0

    def step(self, dt: float, source_fn, dt_sim: float = 0.002) -> np.ndarray:
        """Advance the chain, pinning mass 0 to ``source_fn(time)`` each sub-step."""
        steps = max(1, int(dt / dt_sim) + 1)
        actual_dt = dt / steps
        for _ in range(steps):
            self.time += actual_dt
            self.position[0] = source_fn(self.time)
            self._substep(actual_dt)
        return self.position

    def _substep(self, dt: float) -> None:
        y, v, m, k = self.position, self.velocity, self.mass, self.stiffness
        force = np.zeros_like(y)
        force[1:-1] = k * (y[:-2] - 2 * y[1:-1] + y[2:])
        force[-1] = k * (y[-2] - y[-1])  # free boundary at the far end

        v[1:] += (force[1:] - self.damping * v[1:]) / m[1:] * dt
        y[1:] += v[1:] * dt


def reflection_transmission_coefficients(impedance_1: float, impedance_2: float) -> tuple[float, float]:
    """Normal-incidence reflection and transmission coefficients at an
    acoustic-impedance boundary (``Z = density * velocity``).
    """
    reflection = (impedance_2 - impedance_1) / (impedance_2 + impedance_1)
    transmission = 1.0 + reflection
    return reflection, transmission


# ---------------------------------------------------------------------------
# 2D acoustic wave-equation solver, built on the `findiff` FD library
# ---------------------------------------------------------------------------

@dataclass
class AcousticWaveSolver2D:
    """Explicit finite-difference solver for the 2D scalar acoustic wave
    equation ``u_tt = v^2 * Laplacian(u)``.

    The Laplacian is built once with :mod:`findiff`, a well-tested,
    pure-Python finite-difference operator library, instead of hand-written
    array slicing -- this keeps the numerical scheme declarative and makes
    the accuracy order trivially tunable via ``space_order``.
    """

    nz: int
    nx: int
    dx: float
    dt: float
    velocity: np.ndarray  # (nz, nx) velocity model, meters/second
    space_order: int = 2
    boundary_pad: int = 25
    boundary_min_retention: float = 0.985

    def __post_init__(self) -> None:
        self._laplacian = Diff(0, self.dx, acc=self.space_order) ** 2 + Diff(1, self.dx, acc=self.space_order) ** 2
        self.u_prev = np.zeros((self.nz, self.nx))
        self.u_curr = np.zeros((self.nz, self.nx))
        self.damping_mask = self._build_sponge_mask()
        self.time_step = 0

    def _build_sponge_mask(self) -> np.ndarray:
        """Quadratically tapered absorbing boundary (Cerjan-style sponge)."""
        mask = np.ones((self.nz, self.nx))
        pad = self.boundary_pad
        for i in range(pad):
            taper = 1.0 - (1.0 - self.boundary_min_retention) * ((pad - i) / pad) ** 2
            mask[i, :] *= taper
            mask[-1 - i, :] *= taper
            mask[:, i] *= taper
            mask[:, -1 - i] *= taper
        return mask

    def inject_source(self, row: int, col: int, amplitude: float) -> None:
        self.u_curr[row, col] += amplitude

    def advance(self) -> np.ndarray:
        """Step the wavefield forward by one ``dt`` and return the new state."""
        laplacian = self._laplacian(self.u_curr)
        u_next = 2 * self.u_curr - self.u_prev + (self.velocity * self.dt) ** 2 * laplacian
        u_next *= self.damping_mask
        self.u_prev, self.u_curr = self.u_curr, u_next
        self.time_step += 1
        return self.u_curr


# ---------------------------------------------------------------------------
# Normal moveout (NMO)
# ---------------------------------------------------------------------------

def hyperbolic_traveltime(t0: float, rms_velocity: float, offset) -> np.ndarray:
    """CMP-gather hyperbolic arrival time ``t_x = sqrt(t0^2 + (x / v)^2)``."""
    offset = np.asarray(offset)
    return np.sqrt(t0**2 + (offset / rms_velocity) ** 2)


def nmo_shift(t0: float, rms_velocity: float, offset) -> np.ndarray:
    """Time shift required to flatten a reflection to its zero-offset time ``t0``."""
    return hyperbolic_traveltime(t0, rms_velocity, offset) - t0


# ---------------------------------------------------------------------------
# Predictive deconvolution / Wiener-Hopf prediction error filter (PEF)
# ---------------------------------------------------------------------------

def autocorrelation(trace: np.ndarray, max_lag: int) -> np.ndarray:
    """One-sided (non-negative lags) biased autocorrelation of ``trace``."""
    full = np.correlate(trace, trace, mode="full")
    zero_lag = len(trace) - 1
    return full[zero_lag : zero_lag + max_lag + 1]


def find_dominant_lag(trace: np.ndarray, min_lag: int) -> int:
    """Return the lag (>= ``min_lag`` samples) of peak autocorrelation, i.e.
    the period of a suspected periodic reverberation/multiple in ``trace``."""
    autocorr = autocorrelation(trace, len(trace) - 1)
    return min_lag + int(np.argmax(autocorr[min_lag:]))


def design_pef(autocorr: np.ndarray, filter_length: int, prediction_gap: int) -> np.ndarray:
    """Solve the Wiener-Hopf normal equations ``R f = g`` for a prediction
    error filter, via a Toeplitz solve (Levinson recursion under the hood).

    Args:
        autocorr: Autocorrelation lags ``[r_0, r_1, ...]`` of the input trace.
        filter_length: Number of filter coefficients to solve for.
        prediction_gap: Lag ``alpha`` separating the filter from the primary
            it is built to predict (e.g. the two-way time of a water-layer
            multiple).

    Returns:
        Filter coefficients ``f`` such that the predicted signal at lag
        ``alpha + i`` is ``sum_j f[j] * trace[t - j]``.
    """
    r = autocorr[:filter_length]
    g = autocorr[prediction_gap : prediction_gap + filter_length]
    return solve_toeplitz(r, g)


def predictive_deconvolution(trace: np.ndarray, prediction_gap: int, filter_length: int = 3) -> np.ndarray:
    """Suppress periodic multiples in ``trace`` using a Wiener-Hopf PEF.

    Builds a short prediction filter from the trace's own autocorrelation
    (the classic Peacock & Treitel / Robinson approach), predicts the
    reverberant part of the signal ``prediction_gap`` samples ahead, and
    subtracts it out.
    """
    autocorr = autocorrelation(trace, prediction_gap + filter_length)
    pef = design_pef(autocorr, filter_length, prediction_gap)

    deconvolved = trace.copy()
    for i in range(prediction_gap, len(trace)):
        prediction = sum(pef[j] * deconvolved[i - prediction_gap - j] for j in range(filter_length))
        deconvolved[i] -= prediction
    return deconvolved
