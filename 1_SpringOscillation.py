"""Damped vertical spring-mass oscillator, kicked off by a hammer strike.

y(t) = -A * exp(-gamma t) * cos(omega_d t), triggered at t=0 by an impulsive
hit on top of the mass. This scene is purely kinematic (a closed-form
solution rather than a numerical integrator), so it only needs the shared
zigzag-spring visual helper from ``seismic_lib``.
"""

from manim import *
import numpy as np

from seismic_lib import spring_zigzag_points

REST_LENGTH = 3.0
AMPLITUDE = 1.5
FREQUENCY_HZ = 0.5
OMEGA_0 = 2 * PI * FREQUENCY_HZ
DAMPING_RATE = 0.25  # gamma, 1/s
OMEGA_D = np.sqrt(max(OMEGA_0**2 - DAMPING_RATE**2, 0.0))  # damped angular frequency
ANIMATION_DURATION = 10.0
STRIKE_RUN_TIME = 0.35
RETRACT_RUN_TIME = 0.5


def displacement(t: float) -> float:
    """Damped oscillator displacement from equilibrium at time ``t``."""
    return -AMPLITUDE * np.exp(-DAMPING_RATE * t) * np.cos(OMEGA_D * t)


class SpringOscillation(Scene):
    """A hammer slams a block on a spring, which then oscillates and decays."""

    def construct(self):
        # 1. Establish anchor, physics properties, and visual elements
        # Shifted the ceiling and anchor to the left to make room for the plot
        ceiling = Line(LEFT * 5, LEFT * 1, stroke_width=4).to_edge(UP, buff=1.0)
        anchor_point = ceiling.get_center()

        # Block starts at rest (undisturbed equilibrium), before the strike.
        block = Square(side_length=1.0, fill_color=BLUE_D, fill_opacity=0.8, color=WHITE)
        block.move_to([anchor_point[0], anchor_point[1] - REST_LENGTH - block.height / 2, 0])

        spring = VMobject(stroke_color=GRAY_B, stroke_width=3)
        spring.set_points_as_corners(spring_zigzag_points(anchor_point, block.get_top(), n_coils=12, width=0.4))

        # Hammer: a handle with a mallet head, raised above the block.
        handle = Rectangle(width=0.15, height=1.3, fill_color="#8B5A2B", fill_opacity=1, stroke_width=0)
        head = Rectangle(width=0.9, height=0.4, fill_color=GRAY_D, fill_opacity=1, color=WHITE, stroke_width=1)
        head.next_to(handle, DOWN, buff=-0.05)
        hammer = VGroup(handle, head).next_to(block, UP, buff=1.2)

        # 2. Setup Diagram
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-2, 2, 1],
            x_length=6,
            y_length=4,
            axis_config={"include_numbers": True},
        ).to_corner(UR, buff=1.0)

        labels = axes.get_axis_labels(Text("t", font_size=20), Text("y", font_size=20))

        wave_path = VMobject(stroke_color=YELLOW, stroke_width=3)
        trace_dot = Dot(color=YELLOW)
        link_line = DashedLine(color=GRAY, dash_length=0.1)

        # 3. Dynamic updaters to link the mass, spring, plot, and time
        t_tracker = ValueTracker(0)

        def update_block(mob):
            t = t_tracker.get_value()
            current_y = anchor_point[1] - REST_LENGTH + displacement(t)
            mob.move_to([anchor_point[0], current_y - mob.height / 2, 0])

        def update_spring(mob):
            mob.set_points_as_corners(spring_zigzag_points(anchor_point, block.get_top(), n_coils=12, width=0.4))

        def update_wave(mob):
            t = t_tracker.get_value()
            if t > 0:
                new_curve = axes.plot(displacement, x_range=[0, max(0.001, t)], color=YELLOW)
                mob.become(new_curve)

        def update_trace_dot(mob):
            t = t_tracker.get_value()
            mob.move_to(axes.c2p(t, displacement(t)))

        def update_link_line(mob):
            t = t_tracker.get_value()
            mob.put_start_and_end_on(block.get_right(), axes.c2p(t, displacement(t)))

        # 4. Render static setup, then animate the strike before the decay
        title = Text("Hammer Strike: Damped Spring Oscillation", font_size=28).to_edge(DOWN, buff=1.0)

        self.add(
            ceiling, spring, block, hammer, title,
            axes, labels, wave_path, trace_dot, link_line
        )

        # Swing the hammer down onto the block, then retract it on impact.
        self.play(hammer.animate.next_to(block, UP, buff=0.0), run_time=STRIKE_RUN_TIME, rate_func=rush_into)
        self.play(
            Flash(block.get_top(), color=YELLOW, flash_radius=0.5),
            FadeOut(hammer, shift=UP * 0.8),
            run_time=RETRACT_RUN_TIME,
        )

        # 5. Impact delivered: let the block oscillate and decay
        block.add_updater(update_block)
        spring.add_updater(update_spring)
        wave_path.add_updater(update_wave)
        trace_dot.add_updater(update_trace_dot)
        link_line.add_updater(update_link_line)

        self.play(
            t_tracker.animate.set_value(ANIMATION_DURATION),
            run_time=ANIMATION_DURATION,
            rate_func=linear,
        )