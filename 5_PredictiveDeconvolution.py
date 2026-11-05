"""Predictive deconvolution applied to a synthetic seismic trace.

The trace is generated from the same stiff-layer mass-spring chain used in
``StiffLayerOscillation`` / ``VerticalSeismicTrace``. A Wiener-Hopf
prediction-error filter (``seismic_lib.predictive_deconvolution``, solved
via a Toeplitz/Levinson recursion) then suppresses the reverberant
multiples, side by side with the raw trace.
"""

from manim import *
import numpy as np

from seismic_lib import MassSpringChain1D, find_dominant_lag, predictive_deconvolution

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
MIN_MULTIPLE_LAG_SECONDS = 0.15
PEF_LENGTH = 3


def simulate_top_block_trace() -> tuple[np.ndarray, np.ndarray]:
    """Run the stiff-layer chain and return (times, top-block displacement),
    downsampled 4x for rendering efficiency."""
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

    history = history[::4]
    times = np.array([t for t, _ in history])
    trace = np.array([disp for _, disp in history])
    return times, trace


class PredictiveDeconvolution(Scene):
    """Side-by-side comparison of a raw trace and its deconvolved counterpart."""

    def construct(self):
        # 1. Physics: synthesize the raw trace
        times, raw_trace = simulate_top_block_trace()

        # 2. Predictive Deconvolution: find the multiple's period, then design
        # and apply a Wiener-Hopf prediction-error filter to suppress it.
        dt = times[1] - times[0]
        min_lag = int(MIN_MULTIPLE_LAG_SECONDS / dt)
        lag = find_dominant_lag(raw_trace, min_lag)
        deconvolved_trace = predictive_deconvolution(raw_trace, prediction_gap=lag, filter_length=PEF_LENGTH)

        # 3. Visual Setup (Side-by-Side Comparison)
        def create_axes(position):
            ax = Axes(
                x_range=[-0.2, 0.2, 0.1],
                y_range=[-2.0, 0, 0.5],
                x_length=4,
                y_length=6.5,
                axis_config={"include_numbers": False},
            ).move_to(position)

            y_label_dict = {
                0: MathTex("0.0"), -0.5: MathTex("0.5"), -1.0: MathTex("1.0"),
                -1.5: MathTex("1.5"), -2.0: MathTex("2.0"),
            }
            ax.get_y_axis().add_labels(y_label_dict)

            x_label_dict = {-0.2: MathTex("-0.2"), 0: MathTex("0"), 0.2: MathTex("0.2")}
            ax.get_x_axis().add_labels(x_label_dict)
            return ax

        ax_left = create_axes(LEFT * 3)
        ax_right = create_axes(RIGHT * 3)

        title_left = Text("Original Trace", font_size=24).next_to(ax_left, UP)
        title_right = Text("Predictive Deconvolution", font_size=24, color=YELLOW).next_to(ax_right, UP)

        # Plot Curves
        curve_left = VMobject(stroke_color=WHITE, stroke_width=2)
        curve_left.set_points_as_corners([ax_left.c2p(raw_trace[i], -times[i]) for i in range(len(times))])

        curve_right = VMobject(stroke_color=YELLOW, stroke_width=2)
        curve_right.set_points_as_corners([ax_right.c2p(deconvolved_trace[i], -times[i]) for i in range(len(times))])

        self.play(
            Create(ax_left), Create(ax_right),
            FadeIn(title_left), FadeIn(title_right)
        )
        self.play(Create(curve_left), run_time=2.0)

        # 4. Apply User's Highlight Windows to the Original Trace
        def highlight_packet(ax, t_start, t_end, color, text_label):
            top_left = ax.c2p(-0.2, -t_start)
            bottom_right = ax.c2p(0.2, -t_end)

            box = Rectangle(
                width=abs(bottom_right[0] - top_left[0]),
                height=abs(top_left[1] - bottom_right[1]),
                stroke_width=2,
                stroke_color=color,
                fill_color=color,
                fill_opacity=0.2
            ).move_to(ax.c2p(0, -(t_start + t_end) / 2))

            label = Text(text_label, font_size=16, color=color).next_to(box, LEFT, buff=0.2)
            return VGroup(box, label)

        pulse_highlight = highlight_packet(ax_left, 0.0, 0.18, BLUE, "Source")
        primary_highlight = highlight_packet(ax_left, 0.22, 0.30, GREEN, "Primary")
        multiple_highlight = highlight_packet(ax_left, 0.35, 0.68, RED, "First Multi")
        second_multiple = highlight_packet(ax_left, 0.72, 0.99, ORANGE, "Second Multi")

        self.play(FadeIn(pulse_highlight), FadeIn(primary_highlight))
        self.play(FadeIn(multiple_highlight), FadeIn(second_multiple))
        self.wait(1)

        # Draw the filtered trace
        self.play(Create(curve_right), run_time=2.0)

        # Add a subtle highlight on the right graph to show the multiples are gone
        clean_zone = Rectangle(
            width=ax_right.x_length,
            height=ax_right.c2p(0, -0.35)[1] - ax_right.c2p(0, -2.0)[1],
            stroke_width=0, fill_color=GREEN, fill_opacity=0.1
        ).move_to(ax_right.c2p(0, -1.175))

        clean_label = Text("Multiples Suppressed", font_size=20, color=GREEN).next_to(clean_zone, RIGHT, buff=0.2)

        self.play(FadeIn(clean_zone), FadeIn(clean_label))
        self.wait(4)
        def create_axes(position):
            ax = Axes(
                x_range=[-0.2, 0.2, 0.1],       
                y_range=[-2.0, 0, 0.5], 
                x_length=4,
                y_length=6.5,
                axis_config={"include_numbers": False},
            ).move_to(position)
            
            y_label_dict = {
                0: MathTex("0.0"), -0.5: MathTex("0.5"), -1.0: MathTex("1.0"),
                -1.5: MathTex("1.5"), -2.0: MathTex("2.0")
            }
            ax.get_y_axis().add_labels(y_label_dict)
            
            x_label_dict = {-0.2: MathTex("-0.2"), 0: MathTex("0"), 0.2: MathTex("0.2")}
            ax.get_x_axis().add_labels(x_label_dict)
            return ax

        ax_left = create_axes(LEFT * 3)
        ax_right = create_axes(RIGHT * 3)
        
        title_left = Text("Original Trace", font_size=24).next_to(ax_left, UP)
        title_right = Text("Predictive Deconvolution", font_size=24, color=YELLOW).next_to(ax_right, UP)

        # Plot Curves
        curve_left = VMobject(stroke_color=WHITE, stroke_width=2)
        curve_left.set_points_as_corners([ax_left.c2p(raw_trace[i], -times[i]) for i in range(len(times))])
        
        curve_right = VMobject(stroke_color=YELLOW, stroke_width=2)
        curve_right.set_points_as_corners([ax_right.c2p(deconvolved_trace[i], -times[i]) for i in range(len(times))])

        self.play(
            Create(ax_left), Create(ax_right), 
            FadeIn(title_left), FadeIn(title_right)
        )
        self.play(Create(curve_left), run_time=2.0)

        # 4. Apply User's Highlight Windows to the Original Trace
        def highlight_packet(ax, t_start, t_end, color, text_label):
            top_left = ax.c2p(-0.2, -t_start)
            bottom_right = ax.c2p(0.2, -t_end)
            
            box = Rectangle(
                width=abs(bottom_right[0] - top_left[0]),
                height=abs(top_left[1] - bottom_right[1]), 
                stroke_width=2,
                stroke_color=color,
                fill_color=color,
                fill_opacity=0.2
            ).move_to(ax.c2p(0, -(t_start + t_end) / 2))
            
            label = Text(text_label, font_size=16, color=color).next_to(box, LEFT, buff=0.2)
            return VGroup(box, label)

        pulse_highlight = highlight_packet(ax_left, 0.0, 0.18, BLUE, "Source")
        primary_highlight = highlight_packet(ax_left, 0.22, 0.30, GREEN, "Primary")
        multiple_highlight = highlight_packet(ax_left, 0.35, 0.68, RED, "First Multi")
        second_multiple = highlight_packet(ax_left, 0.72, 0.99, ORANGE, "Second Multi")

        self.play(FadeIn(pulse_highlight), FadeIn(primary_highlight))
        self.play(FadeIn(multiple_highlight), FadeIn(second_multiple))
        self.wait(1)

        # Draw the filtered trace
        self.play(Create(curve_right), run_time=2.0)
        
        # Add a subtle highlight on the right graph to show the multiples are gone
        clean_zone = Rectangle(
            width=ax_right.x_length, 
            height=ax_right.c2p(0, -0.35)[1] - ax_right.c2p(0, -2.0)[1],
            stroke_width=0, fill_color=GREEN, fill_opacity=0.1
        ).move_to(ax_right.c2p(0, -1.175))
        
        clean_label = Text("Multiples Suppressed", font_size=20, color=GREEN).next_to(clean_zone, RIGHT, buff=0.2)
        
        self.play(FadeIn(clean_zone), FadeIn(clean_label))
        self.wait(4)