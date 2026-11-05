"""Normal moveout (NMO) correction: flattening hyperbolic reflection events
in a common-midpoint gather so they align at their true zero-offset time.

The traces animated here are not synthesized: they are the real surface
receiver recordings from the same 2D mass-spring shot-gather physics used in
``Grid2DOscillation.py`` (``seismic_lib.MassSpringGrid2D``), so the hyperbola
being flattened is the actual reflection off that scene's stiff layer. The
reflector's zero-offset time and overburden velocity are derived analytically
from that same chain's stiffness/mass/length (continuum-limit wave speed),
not fitted or guessed.
"""

from manim import *
import numpy as np

from seismic_lib import MassSpringGrid2D, hyperbolic_traveltime, nmo_shift

# Shot-gather physics: identical setup to Grid2DOscillation.py, so the traces
# recorded below are the real receiver signals from that same model.
GRID_N_ROWS = 50
GRID_N_COLS = 101
GRID_TOTAL_LENGTH = 6.0
GRID_TOTAL_MASS = 5.0
GRID_BASE_STIFFNESS_Y = 30.0
GRID_BASE_STIFFNESS_X = 5.0
GRID_STIFF_BAND = (0.40, 0.76)
GRID_STIFF_MULTIPLIER = 8.0
GRID_SPONGE_START_FRAC = 0.76
GRID_SPONGE_MAX_DAMPING = 25.0
GRID_SPACING_X = 0.12
GRID_ANCHOR_Y = 3.5
GRID_PULSE_ROW = 5
GRID_PULSE_ROW_WIDTH = GRID_N_ROWS * 0.05
GRID_PULSE_COL_WIDTH = GRID_N_COLS * 0.05
GRID_SIM_TIME_STEP = 0.002
GRID_RECEIVER_COLUMNS = [GRID_N_COLS // 2 + 5 * i for i in range(1, 11)]

RECORD_DURATION = 0.9  # sim-time seconds to capture the reflection at every offset
RECORD_STEPS = 1500
WAVELET_AMPLITUDE_SCALE = 2.0
TRACE_TIME_SAMPLES = np.linspace(0.0, RECORD_DURATION, 300)


def _simulate_shot_gather_receivers():
    """Run the same shot-gather physics as Grid2DOscillation and record each
    surface receiver's (times, amplitude) trace, plus the reflector's analytic
    (t0, velocity) derived from the chain's continuum-limit wave speed.
    """
    grid = MassSpringGrid2D(
        n_rows=GRID_N_ROWS,
        n_cols=GRID_N_COLS,
        total_length=GRID_TOTAL_LENGTH,
        total_mass=GRID_TOTAL_MASS,
        base_stiffness_y=GRID_BASE_STIFFNESS_Y,
        base_stiffness_x=GRID_BASE_STIFFNESS_X,
        anchor_y=GRID_ANCHOR_Y,
        stiff_band=GRID_STIFF_BAND,
        stiff_multiplier=GRID_STIFF_MULTIPLIER,
        sponge_start_frac=GRID_SPONGE_START_FRAC,
        sponge_max_damping=GRID_SPONGE_MAX_DAMPING,
    )
    center_col = GRID_N_COLS // 2
    grid.apply_gaussian_pulse(GRID_PULSE_ROW, center_col, GRID_PULSE_ROW_WIDTH, GRID_PULSE_COL_WIDTH, amplitude=1.0)

    times = np.empty(RECORD_STEPS)
    receiver_amplitude = {col: np.empty(RECORD_STEPS) for col in GRID_RECEIVER_COLUMNS}
    dt_record = RECORD_DURATION / RECORD_STEPS
    for step in range(RECORD_STEPS):
        grid.step(dt_record, dt_sim=GRID_SIM_TIME_STEP)
        times[step] = grid.time
        displacement = grid.displacement()
        for col in GRID_RECEIVER_COLUMNS:
            receiver_amplitude[col][step] = displacement[0, col]

    offsets = np.array([(col - center_col) * GRID_SPACING_X for col in GRID_RECEIVER_COLUMNS])

    # Continuum-limit wave speed of the overburden, from the chain's own
    # stiffness/mass/length: c = total_length * sqrt(k / mass) (independent
    # of the discretization n_rows). The reflector sits at the top of the
    # stiff band, at depth stiff_band[0] * total_length.
    velocity = GRID_TOTAL_LENGTH * np.sqrt(GRID_BASE_STIFFNESS_Y / GRID_TOTAL_MASS)
    reflector_depth = GRID_STIFF_BAND[0] * GRID_TOTAL_LENGTH
    t0 = 2 * reflector_depth / velocity

    return times, receiver_amplitude, offsets, t0, velocity


TRACE_TIMES, TRACE_AMPLITUDE, OFFSETS, REFLECTOR_T0, REFLECTOR_VELOCITY = _simulate_shot_gather_receivers()


class NormalMoveoutCorrection(Scene):
    """Animates NMO-flattening of the real receiver traces from the shot-gather scene."""

    def construct(self):
        # 1. UI and Equation Setup
        title = Text("Normal Moveout (NMO) Correction", font_size=32).to_edge(UP, buff=0.3)

        eq_text = Text("Hyperbolic Arrival Time:", font_size=18, color=BLUE)
        eq_math = MathTex("t_x = \\sqrt{t_0^2 + \\frac{x^2}{v^2}}", font_size=28)

        nmo_eq_text = Text("Correction Shift:", font_size=18, color=YELLOW)
        nmo_eq_math = MathTex("\\Delta t = t_x - t_0", font_size=28)

        eq_group = VGroup(eq_text, eq_math, nmo_eq_text, nmo_eq_math).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        eq_group.to_corner(UL, buff=0.8).shift(DOWN * 0.5)

        self.play(FadeIn(title), FadeIn(eq_group))

        # 2. Axes Setup (matched to the real recorded offset/time ranges)
        axes = Axes(
            x_range=[0, 6.5, 1],
            y_range=[-(RECORD_DURATION + 0.1), 0, 0.3],
            x_length=6.5,
            y_length=6.0,
            axis_config={"include_numbers": True}
        ).next_to(title, DOWN, buff=0.3).shift(RIGHT * 2.0)

        axis_labels = axes.get_axis_labels(Text("Offset x", font_size=16), Text("Time t (s)", font_size=16))

        self.play(Create(axes), FadeIn(axis_labels))

        # 3. Reference line marking the reflector's zero-offset time
        ref_line = DashedLine(
            start=axes.c2p(0, -REFLECTOR_T0), end=axes.c2p(6.5, -REFLECTOR_T0), color=ORANGE, dash_length=0.1
        )
        ref_label = MathTex(f"t_0 = {REFLECTOR_T0:.2f}\\text{{s}}", font_size=18, color=ORANGE).next_to(
            ref_line, RIGHT, buff=0.1
        )
        self.play(Create(VGroup(ref_line, ref_label)))

        # 4. Real traces recorded from the shot-gather scene's receivers
        correction_tracker = ValueTracker(0.0)
        traces = VGroup(*[VMobject(stroke_width=1.5, stroke_color=WHITE) for _ in OFFSETS])

        def update_traces(mob):
            ratio = correction_tracker.get_value()
            for i, x in enumerate(OFFSETS):
                shift = ratio * nmo_shift(REFLECTOR_T0, REFLECTOR_VELOCITY, x)
                amplitude = np.interp(
                    TRACE_TIME_SAMPLES + shift,
                    TRACE_TIMES,
                    TRACE_AMPLITUDE[GRID_RECEIVER_COLUMNS[i]],
                    left=0.0,
                    right=0.0,
                ) * WAVELET_AMPLITUDE_SCALE
                points = [axes.c2p(x + a, -t) for t, a in zip(TRACE_TIME_SAMPLES, amplitude)]
                mob[i].set_points_as_corners(points)

        traces.add_updater(update_traces)

        self.play(Create(traces), run_time=2.0)
        self.wait(1)

        # 5. Apply the NMO Correction
        status_text = Text("Applying NMO Time Shift...", font_size=18, color=YELLOW)
        status_text.next_to(eq_group, DOWN, buff=1.0, aligned_edge=LEFT)
        self.play(FadeIn(status_text))

        self.play(correction_tracker.animate.set_value(1.0), run_time=4.0, rate_func=smooth)

        done_text = Text("Gathers Flattened", font_size=18, color=GREEN).move_to(status_text, aligned_edge=LEFT)
        self.play(Transform(status_text, done_text))

        # Add a vertical stacking highlight
        stack_highlight = Rectangle(
            width=0.4, height=axes.y_length,
            stroke_width=2, stroke_color=BLUE,
            fill_color=BLUE, fill_opacity=0.2
        ).move_to(axes.c2p(3.3, -(RECORD_DURATION / 2)))

        stack_label = Text("Ready to Stack", font_size=16, color=BLUE).next_to(stack_highlight, DOWN, buff=0.1)

        self.play(FadeIn(stack_highlight), FadeIn(stack_label))
        self.wait(3)