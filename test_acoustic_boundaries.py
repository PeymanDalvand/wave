"""Numerical regression checks for the acoustic solver's absorbing edges."""

import unittest

import numpy as np

from seismic_lib import AcousticWaveSolver2D, ricker_wavelet


class AcousticBoundaryTests(unittest.TestCase):
    def test_reflection_against_larger_domain(self):
        solver = AcousticWaveSolver2D(
            100, 100, 10.0, 0.001, np.full((100, 100), 2000.0), boundary_pad=20
        )
        reference = AcousticWaveSolver2D(
            260, 260, 10.0, 0.001, np.full((260, 260), 2000.0), boundary_pad=0
        )
        peak_reference = 0.0
        peak_error = 0.0
        for step in range(650):
            pulse = float(ricker_wavelet(step * solver.dt, 15.0))
            solver.inject_source(50, 50, pulse)
            reference.inject_source(130, 130, pulse)
            solver.advance()
            reference.advance()
            reference_window = reference.u_curr[105:155, 105:155]
            peak_reference = max(peak_reference, float(np.linalg.norm(reference_window)))
            if step >= 350:
                error = solver.u_curr[25:75, 25:75] - reference_window
                peak_error = max(peak_error, float(np.linalg.norm(error)))

        self.assertLess(peak_error / peak_reference, 0.06)

    def test_all_four_sides_absorb(self):
        for side, source in (
            ("top", (30, 50)),
            ("bottom", (69, 50)),
            ("left", (50, 30)),
            ("right", (50, 69)),
        ):
            with self.subTest(side=side):
                solver = AcousticWaveSolver2D(
                    100, 100, 10.0, 0.001, np.full((100, 100), 2000.0), boundary_pad=20
                )
                peak_norm = 0.0
                for step in range(1000):
                    pulse = float(ricker_wavelet(step * solver.dt, 15.0))
                    solver.inject_source(*source, pulse)
                    solver.advance()
                    peak_norm = max(peak_norm, float(np.linalg.norm(solver.u_curr)))

                self.assertTrue(np.isfinite(solver.u_curr).all())
                self.assertLess(float(np.linalg.norm(solver.u_curr)) / peak_norm, 0.01)

    def test_layered_model_stays_stable(self):
        velocity = np.full((200, 200), 2000.0)
        velocity[100:150] = 4000.0
        velocity[150:] = 5000.0
        solver = AcousticWaveSolver2D(200, 200, 10.0, 0.001, velocity, boundary_pad=25)
        for step in range(3000):
            solver.inject_source(50, 50, 0.5 * float(ricker_wavelet(step * solver.dt, 15.0)))
            solver.advance()

        self.assertEqual(solver.time_step, 3000)
        self.assertTrue(np.isfinite(solver.u_curr).all())
        self.assertLess(float(np.max(np.abs(solver.u_curr))), 0.001)


if __name__ == "__main__":
    unittest.main()