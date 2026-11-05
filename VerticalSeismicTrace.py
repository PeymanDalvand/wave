"""Vertical seismic trace: the top-block displacement of a stiff-layer chain,
plotted as a classic time-downward wiggle trace with annotated arrivals.

Physics (identical model to ``StiffLayerOscillation``) is precomputed
offline via ``seismic_lib.MassSpringChain1D`` before any animation starts.
"""

from manim import *
import numpy as np

from seismic_lib import MassSpringChain1D

NUM_BLOCKS = 130
TOTAL_LENGTH = 7.0
TOTAL_MASS = 5.0
BASE_STIFFNESS = 30.0
STIFF_BAND = (0.40, 0.76)
STIFF_MULTIPLIER = 8.0
SPONGE_START_FRAC = 0.76
SPONGE_MAX_DAMPING = 25.0
PULSE_CENTER = NUM_BLOCKS * 0.1
PULSE_WIDTH = NUM_BLOCKS * 0.04
SIM_TIME_STEP = 0.001
SIM_DURATION = 2.0
ANCHOR_Y = 3.5


def simulate_top_block_trace() -> list[tuple[float, float]]:
    """Run the stiff-layer chain for ``SIM_DURATION`` seconds and record the
    top block's displacement, downsampled 4x for rendering efficiency."""
    chain = MassSpringChain1D(
        n=NUM_BLOCKS,
        total_length=TOTAL_LENGTH,
        total_mass=TOTAL_MASS,
        base_stiffness=BASE_STIFFNESS,
        anchor_y=ANCHOR_Y,
        stiff_band=STIFF_BAND,
        stiff_multiplier=STIFF_MULTIPLIER,
        sponge_start_frac=SPONGE_START_FRAC,
        sponge_max_damping=SPONGE_MAX_DAMPING,
    )
    chain.apply_gaussian_pulse(PULSE_CENTER, PULSE_WIDTH, amplitude=0.6)

    history = []
    while chain.time <= SIM_DURATION:
        chain.advance(SIM_TIME_STEP)
        history.append((chain.time, chain.displacement()[0]))
    return history[::4]


class VerticalSeismicTrace(Scene):
    """Renders the precomputed top-block trace as a time-downward wiggle plot."""

    def construct(self):
        history = simulate_top_block_trace()

        # 1. Visual Setup for the Vertical Trace (First 2 Seconds)
        axes = Axes(
            x_range=[-0.2, 0.2, 0.1],
            y_range=[-2.0, 0, 0.5],  # 2 physical seconds, inverted
            x_length=6,
            y_length=7,
            axis_config={"include_numbers": False},
        ).to_edge(LEFT, buff=1.0)

        # Manually assign positive labels to the negative coordinates
        y_label_dict = {
            0: MathTex("0.0"),
            -0.5: MathTex("0.5"),
            -1.0: MathTex("1.0"),
            -1.5: MathTex("1.5"),
            -2.0: MathTex("2.0"),
        }
        axes.get_y_axis().add_labels(y_label_dict)

        x_label_dict = {
            -0.2: MathTex("-0.2"),
            -0.1: MathTex("-0.1"),
            0: MathTex("0"),
            0.1: MathTex("0.1"),
            0.2: MathTex("0.2"),
        }
        axes.get_x_axis().add_labels(x_label_dict)

        axes_labels = axes.get_axis_labels(Text("disp", font_size=20), Text("t (s)", font_size=20))

        # Plot the data: mapping positive time to the negative y-axis coordinates
        t_points = [axes.c2p(disp, -t) for t, disp in history]
        trace_curve = VMobject(stroke_color=YELLOW, stroke_width=2)
        trace_curve.set_points_as_corners(t_points)

        title = Text("Vertical Seismic Trace (First 2 Seconds)", font_size=32).to_edge(UP)
        self.play(FadeIn(title), Create(axes), FadeIn(axes_labels))

        # Draw the trace much slower so we can watch it form
        self.play(Create(trace_curve), run_time=4.0, rate_func=linear)

        # 2. Highlighting the Wave Packets based on exact physics timing
        def highlight_packet(t_start, t_end, color, text_label):
            top_left = axes.c2p(-0.2, -t_start)
            bottom_right = axes.c2p(0.2, -t_end)

            box = Rectangle(
                width=abs(bottom_right[0] - top_left[0]),
                height=abs(top_left[1] - bottom_right[1]),
                stroke_width=2,
                stroke_color=color,
                fill_color=color,
                fill_opacity=0.2,
            ).move_to(axes.c2p(0, -(t_start + t_end) / 2))

            label = Text(text_label, font_size=20, color=color).next_to(box, RIGHT, buff=0.5)
            return VGroup(box, label)

        # Time windows tuned to match the chain's physical wave speed
        pulse_highlight = highlight_packet(0.0, 0.18, BLUE, "Initial Source Pulse")
        primary_highlight = highlight_packet(0.22, 0.30, GREEN, "Primary Reflection")
        multiple_highlight = highlight_packet(0.35, 0.68, RED, "First Multiple")
        second_multiple = highlight_packet(0.72, 0.99, ORANGE, "Second Multiple")

        self.play(FadeIn(pulse_highlight))
        self.wait(0.5)
        self.play(FadeIn(primary_highlight))
        self.wait(0.5)
        self.play(FadeIn(multiple_highlight))
        self.wait(0.5)
        self.play(FadeIn(second_multiple))
        self.wait(4)