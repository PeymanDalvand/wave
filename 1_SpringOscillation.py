"""Damped vertical spring-mass oscillator, kicked off by a hammer strike.

y(t) = -A * exp(-gamma t) * cos(omega_d t), triggered at t=0 by an impulsive
hit on top of the mass. This scene is purely kinematic (a closed-form
solution rather than a numerical integrator), so it only needs the shared
zigzag-spring visual helper from ``seismic_lib``.
"""

from manim import *
import numpy as np

from seismic_lib import spring_zigzag_points

REST_LENGTH = 1.0
AMPLITUDE = 1.5
FREQUENCY_HZ = 0.5
OMEGA_0 = 2 * PI * FREQUENCY_HZ
DAMPING_RATE = 0.25  # gamma, 1/s
OMEGA_D = np.sqrt(max(OMEGA_0**2 - DAMPING_RATE**2, 0.0))  # damped angular frequency
MASS = 1.0
SPRING_CONSTANT = MASS * OMEGA_0**2
DAMPING_COEFFICIENT = 2 * MASS * DAMPING_RATE
ANIMATION_DURATION = 16.0
STRIKE_RUN_TIME = 0.35
RETRACT_RUN_TIME = 0.5


def displacement(t: float) -> float:
    """Damped oscillator displacement from equilibrium at time ``t``."""
    return -AMPLITUDE * np.exp(-DAMPING_RATE * t) * np.cos(OMEGA_D * t)


def velocity(t: float) -> float:
    """Velocity corresponding to :func:`displacement`."""
    phase = OMEGA_D * t
    return AMPLITUDE * np.exp(-DAMPING_RATE * t) * (
        DAMPING_RATE * np.cos(phase) + OMEGA_D * np.sin(phase)
    )


class SpringOscillation(Scene):
    """A hammer slams a block on a spring, which then oscillates and decays."""

    def construct(self):
        # 1. Establish anchor, physics properties, and visual elements
        # The oscillator will be aligned with the plot's vertical axis below.
        ceiling = Line(LEFT * 0.65, RIGHT * 0.65, stroke_width=4).move_to([-1.1, 3.15, 0])
        anchor_point = ceiling.get_center()

        # Block starts at rest (undisturbed equilibrium), before the strike.
        block = Square(side_length=0.9, fill_color=TEAL_D, fill_opacity=0.95, color=WHITE)
        block.move_to([anchor_point[0], anchor_point[1] - REST_LENGTH - block.height / 2, 0])
        mass_label = MathTex(r"m", font_size=30).move_to(block)

        spring = VMobject(stroke_color=GRAY_B, stroke_width=3)
        spring.set_points_as_corners(spring_zigzag_points(anchor_point, block.get_top(), n_coils=12, width=0.28))

        # Hammer: a handle with a mallet head, raised above the block.
        handle = Rectangle(width=0.15, height=1.3, fill_color="#8B5A2B", fill_opacity=1, stroke_width=0)
        head = Rectangle(width=0.9, height=0.4, fill_color=GRAY_D, fill_opacity=1, color=WHITE, stroke_width=1)
        head.next_to(handle, DOWN, buff=-0.05)
        hammer = VGroup(handle, head).scale(0.78)

        # 2. Setup Diagram
        axes = Axes(
            x_range=[0, ANIMATION_DURATION, 2],
            y_range=[-2, 2, 1],
            x_length=6.0,
            y_length=3.0,
            axis_config={"include_numbers": True, "font_size": 18},
        )
        axes.shift(np.array([0.55, 1.2, 0]) - axes.c2p(0, 0))

        labels = axes.get_axis_labels(Text("t", font_size=20), Text("y", font_size=20))
        axes.y_axis.set_opacity(0)

        origin_point = axes.c2p(0, 0)
        ceiling.move_to(origin_point + UP * (REST_LENGTH + block.height / 2))
        anchor_point = ceiling.get_center()
        block.move_to(origin_point)
        spring.set_points_as_corners(
            spring_zigzag_points(anchor_point, block.get_top(), n_coils=12, width=0.28)
        )
        hammer.next_to(block, LEFT, buff=0.75).shift(UP * 0.55)

        wave_path = VMobject(stroke_color=YELLOW, stroke_width=3)
        trace_dot = Dot(color=YELLOW).set_opacity(0)

        lesson_title = Text("HOW AN OSCILLATION BECOMES A WAVE", font_size=17, color=TEAL_A)
        lesson_title.move_to([-4.15, 1.55, 0])

        newton_equation = MathTex(r"F", "=", r"m", r"a", font_size=44)
        newton_equation[0].set_color(RED_C)
        newton_equation[2].set_color(YELLOW)
        newton_equation[3].set_color(BLUE_B)
        newton_equation.move_to([-4.15, 0.75, 0])
        newton_note = Text("The spring pulls. The mass accelerates.", font_size=20)
        newton_note.move_to([-4.15, 0.1, 0])

        coupling_story = VGroup(
            Text("One mass oscillates.", font_size=23),
            Text("Connected masses pull their neighbors.", font_size=20),
            Text("The motion travels: that is a wave.", font_size=21, color=YELLOW),
        ).arrange(DOWN, buff=0.28).move_to([-4.15, 0.45, 0])

        wave_equation = MathTex(
            r"\frac{\partial^2u}{\partial t^2}",
            r"=",
            r"c^2",
            r"\frac{\partial^2u}{\partial x^2}",
            font_size=39,
        ).move_to([-4.15, 0.55, 0])
        wave_equation[0].set_color(BLUE_B)
        wave_equation[2].set_color(YELLOW)
        wave_equation[3].set_color(RED_C)
        time_caption = Text("acceleration", font_size=16, color=BLUE_B)
        speed_caption = Text("wave speed", font_size=16, color=YELLOW)
        space_caption = Text("neighbor pull", font_size=16, color=RED_C)
        equation_captions = VGroup(time_caption, speed_caption, space_caption)
        equation_captions.arrange(RIGHT, buff=0.42).move_to([-4.15, -0.25, 0])
        plain_language = Text(
            "motion in time  =  wave speed × pull from neighbors",
            font_size=18,
            color=GRAY_A,
        ).move_to([-3.85, -1.0, 0])
        speed_note = Text(
            "c tells us how fast the wave travels through the material.",
            font_size=17,
            color=GRAY_B,
        ).move_to([-3.85, -1.48, 0])

        # 3. Dynamic updaters to link the mass, spring, plot, and time
        t_tracker = ValueTracker(0)

        def update_block(mob):
            t = t_tracker.get_value()
            current_y = anchor_point[1] - REST_LENGTH + displacement(t)
            mob.move_to([anchor_point[0], current_y - mob.height / 2, 0])

        def update_spring(mob):
            mob.set_points_as_corners(spring_zigzag_points(anchor_point, block.get_top(), n_coils=12, width=0.28))

        def update_wave(mob):
            t = t_tracker.get_value()
            if t > 0:
                new_curve = axes.plot(displacement, x_range=[0, max(0.001, t)], color=YELLOW)
                mob.become(new_curve)

        def update_trace_dot(mob):
            t = t_tracker.get_value()
            mob.move_to(axes.c2p(t, displacement(t)))

        def update_mass_label(mob):
            mob.move_to(block)

        def force_arrow_points(x_offset, force, maximum_force):
            start = block.get_center() + RIGHT * x_offset
            length = 0.15 + 0.7 * min(abs(force) / maximum_force, 1.0)
            direction = UP if force >= 0 else DOWN
            return start, start + direction * length

        spring_force = -SPRING_CONSTANT * displacement(0)
        damping_force = -DAMPING_COEFFICIENT * velocity(0)
        spring_start, spring_end = force_arrow_points(-0.65, spring_force, SPRING_CONSTANT * AMPLITUDE)
        damping_start, damping_end = force_arrow_points(
            0.65, damping_force, DAMPING_COEFFICIENT * AMPLITUDE * OMEGA_0
        )
        spring_force_arrow = Arrow(spring_start, spring_end, buff=0, color=RED_C, stroke_width=5)
        damping_force_arrow = Arrow(damping_start, damping_end, buff=0, color=BLUE_C, stroke_width=5)
        spring_force_label = MathTex(r"F_s=-ky", font_size=24, color=RED_C)
        damping_force_label = MathTex(r"F_d=-b\dot y", font_size=24, color=BLUE_C)

        def update_spring_force(mob):
            force = -SPRING_CONSTANT * displacement(t_tracker.get_value())
            start, end = force_arrow_points(-0.65, force, SPRING_CONSTANT * AMPLITUDE)
            mob.put_start_and_end_on(start, end)

        def update_damping_force(mob):
            force = -DAMPING_COEFFICIENT * velocity(t_tracker.get_value())
            start, end = force_arrow_points(
                0.65, force, DAMPING_COEFFICIENT * AMPLITUDE * OMEGA_0
            )
            mob.put_start_and_end_on(start, end)

        def update_spring_force_label(mob):
            mob.next_to(block, LEFT, buff=0.35).shift(UP * 0.2)

        def update_damping_force_label(mob):
            mob.next_to(block, LEFT, buff=0.35).shift(DOWN * 0.2)

        # 4. Render static setup, then animate the strike before the decay
        title = Text("DAMPED OSCILLATOR", font_size=30)
        title.to_corner(UL, buff=0.35)
        subtitle = Text("Impulse  →  motion  →  wave model", font_size=17, color=GRAY_B)
        subtitle.next_to(title, DOWN, buff=0.12, aligned_edge=LEFT)

        self.add(
            axes, labels, wave_path, trace_dot,
            ceiling, block, spring, mass_label, hammer, title, subtitle
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
        mass_label.add_updater(update_mass_label)
        wave_path.add_updater(update_wave)
        trace_dot.add_updater(update_trace_dot)

        update_block(block)
        update_spring(spring)
        update_mass_label(mass_label)
        trace_dot.set_opacity(1)
        update_trace_dot(trace_dot)
        update_spring_force(spring_force_arrow)
        update_damping_force(damping_force_arrow)
        update_spring_force_label(spring_force_label)
        update_damping_force_label(damping_force_label)
        spring_force_arrow.add_updater(update_spring_force)
        damping_force_arrow.add_updater(update_damping_force)
        spring_force_label.add_updater(update_spring_force_label)
        damping_force_label.add_updater(update_damping_force_label)
        self.play(
            FadeIn(spring_force_arrow), FadeIn(damping_force_arrow),
            FadeIn(spring_force_label), FadeIn(damping_force_label),
            run_time=0.8,
        )

        self.play(
            t_tracker.animate.set_value(3.0),
            FadeIn(lesson_title), Write(newton_equation), FadeIn(newton_note),
            run_time=3.0,
            rate_func=linear,
        )
        self.play(
            t_tracker.animate.set_value(7.0),
            FadeOut(newton_equation), FadeOut(newton_note), FadeIn(coupling_story),
            run_time=4.0,
            rate_func=linear,
        )
        self.play(
            t_tracker.animate.set_value(12.0),
            FadeOut(coupling_story), Write(wave_equation), FadeIn(equation_captions),
            FadeIn(plain_language), FadeIn(speed_note),
            run_time=5.0,
            rate_func=linear,
        )
        self.play(
            t_tracker.animate.set_value(ANIMATION_DURATION),
            run_time=4.0,
            rate_func=linear,
        )
