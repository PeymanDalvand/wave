"""Two masses hanging in series from springs, kicked off by a hammer strike.

The physics is delegated to ``seismic_lib.MassSpringChain1D`` with ``n=2``,
the same lumped-parameter chain model used by the larger multi-block scenes
in this repository, so the equations of motion are defined and tested in
exactly one place. A hammer strikes the top block to trigger the motion,
matching the hammer-trigger pattern used in ``SpringOscillation.py``.
"""

from manim import *
import numpy as np

from seismic_lib import MassSpringChain1D, spring_zigzag_points

MASS = 1.0
SPRING_CONSTANT = 12.0
REST_LENGTH = 2.0
GRAVITY = 4.0
DAMPING = 0.15
SIM_TIME_STEP = 0.005
STRIKE_RUN_TIME = 0.35
RETRACT_RUN_TIME = 0.5


class CoupledOscillation(Scene):
    """A hammer strikes the top block of a 2-mass chain, then both blocks oscillate."""

    def construct(self):
        # 1. Visual Anchors
        ceiling = Line(LEFT * 5, LEFT * 1, stroke_width=4).to_edge(UP, buff=0.5)
        anchor_point = ceiling.get_center()

        block1 = Square(side_length=0.8, fill_color=BLUE_D, fill_opacity=0.8, color=WHITE)
        block2 = Square(side_length=0.8, fill_color=RED_D, fill_opacity=0.8, color=WHITE)

        # 2. Physics: a 2-mass chain, starting at rest; the hammer strike below
        # is what imparts the initial displacement that drives the motion.
        chain = MassSpringChain1D(
            n=2,
            total_length=2 * REST_LENGTH,
            total_mass=2 * MASS,
            base_stiffness=SPRING_CONSTANT / 2,
            gravity=GRAVITY,
            anchor_y=anchor_point[1],
            base_damping=DAMPING,
        )
        block1.move_to([anchor_point[0], chain.position[0], 0])
        block2.move_to([anchor_point[0], chain.position[1], 0])

        # 3. Springs, Tracing Visuals, & Equations
        spring1 = VMobject(stroke_color=GRAY_B, stroke_width=3)
        spring1.set_points_as_corners(spring_zigzag_points(anchor_point, block1.get_top()))
        spring2 = VMobject(stroke_color=GRAY_B, stroke_width=3)
        spring2.set_points_as_corners(spring_zigzag_points(block1.get_bottom(), block2.get_top()))

        # Hammer: a handle with a mallet head, raised above the top block.
        handle = Rectangle(width=0.15, height=1.0, fill_color="#8B5A2B", fill_opacity=1, stroke_width=0)
        head = Rectangle(width=0.7, height=0.3, fill_color=GRAY_D, fill_opacity=1, color=WHITE, stroke_width=1)
        head.next_to(handle, DOWN, buff=-0.05)
        hammer = VGroup(handle, head).next_to(block1, UP, buff=1.0)

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[-4, 4, 1],
            x_length=7,
            y_length=5,
            axis_config={"include_numbers": True},
        ).to_edge(RIGHT, buff=0.5).shift(DOWN * 0.5)

        labels = axes.get_axis_labels(Text("t", font_size=20), Text("y (disp)", font_size=20))

        trace1 = VMobject(stroke_color=BLUE_D, stroke_width=3)
        trace2 = VMobject(stroke_color=RED_D, stroke_width=3)

        link_line1 = DashedLine(color=BLUE_D, dash_length=0.1, stroke_opacity=0.5)
        link_line2 = DashedLine(color=RED_D, dash_length=0.1, stroke_opacity=0.5)

        # 4. The Physics Engine
        history: list[tuple[float, float, float]] = []

        def step_physics(mob, dt):
            chain.step(dt, dt_sim=SIM_TIME_STEP)
            disp1, disp2 = chain.displacement()
            history.append((chain.time, disp1, disp2))
            mob.move_to([anchor_point[0], chain.position[0], 0])

        def update_traces(m):
            if len(history) > 1:
                t_points1 = [axes.c2p(h[0], h[1]) for h in history]
                t_points2 = [axes.c2p(h[0], h[2]) for h in history]
                trace1.set_points_as_corners(t_points1)
                trace2.set_points_as_corners(t_points2)

                link_line1.put_start_and_end_on(block1.get_right(), t_points1[-1])
                link_line2.put_start_and_end_on(block2.get_right(), t_points2[-1])

        trace1.add_updater(update_traces)

        # 6. Render static setup, then animate the strike before the motion
        title = Text("Numerical Coupled Oscillation", font_size=32).to_edge(UP, buff=0.2)

        self.add(
            ceiling, spring1, spring2, block1, block2, hammer, title,
            axes, labels, trace1, trace2, link_line1, link_line2,
        )

        # Swing the hammer down onto the top block, then retract it on impact.
        self.play(hammer.animate.next_to(block1, UP, buff=0.0), run_time=STRIKE_RUN_TIME, rate_func=rush_into)
        self.play(
            Flash(block1.get_top(), color=YELLOW, flash_radius=0.4),
            FadeOut(hammer, shift=UP * 0.8),
            run_time=RETRACT_RUN_TIME,
        )

        # Impact delivered: this initial displacement is the "energy" the hammer imparted.
        chain.position[0] -= 1.5
        chain.position[1] -= 2.5

        block1.add_updater(step_physics)
        block2.add_updater(lambda m: m.move_to([anchor_point[0], chain.position[1], 0]))
        spring1.add_updater(lambda m: m.set_points_as_corners(spring_zigzag_points(anchor_point, block1.get_top())))
        spring2.add_updater(lambda m: m.set_points_as_corners(spring_zigzag_points(block1.get_bottom(), block2.get_top())))

        self.wait(10)