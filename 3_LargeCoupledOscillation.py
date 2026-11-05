"""A 100-block vertical mass-spring chain with an absorbing (sponge) boundary.

Demonstrates how a lumped-parameter chain, driven by an initial Gaussian
pulse, approximates 1D wave propagation with a clean one-way absorbing
boundary at its far end. Physics lives in ``seismic_lib.MassSpringChain1D``.
Opens with a static close-up of a 2-mass spring picture that zooms out to
reveal it was just the top of the full chain.
"""

from manim import *
import numpy as np

from seismic_lib import MassSpringChain1D, spring_zigzag_points

NUM_BLOCKS = 100
TOTAL_LENGTH = 0.05 * NUM_BLOCKS
TOTAL_MASS = 0.05 * NUM_BLOCKS
BASE_STIFFNESS = 15.0
SPONGE_PAD = 25
SPONGE_START_FRAC = (NUM_BLOCKS - SPONGE_PAD) / NUM_BLOCKS
SIM_TIME_STEP = 0.001
TIME_SCALE = 0.06  # slow-motion factor applied to real animation seconds
PULSE_CENTER = 10
PULSE_WIDTH = 4.0
ANCHOR_POINT = np.array([0, 3.5, 0])
BLOCK_SIDE = 0.03

# Intro close-up: a static 2-mass spring picture (mirrors CoupledOscillation.py)
INTRO_MASS = 1.0
INTRO_SPRING_CONSTANT = 12.0
INTRO_REST_LENGTH = 2.0
INTRO_GRAVITY = 4.0
INTRO_DAMPING = 0.15
INTRO_HOLD_TIME = 1.5
ZOOM_OUT_RUN_TIME = 2.0


class LargeCoupledOscillation(Scene):
    """100-block chain with a sponge boundary and a live displacement trace."""

    def construct(self):
        self._play_intro()

        # 1. Physics
        chain = MassSpringChain1D(
            n=NUM_BLOCKS,
            total_length=TOTAL_LENGTH,
            total_mass=TOTAL_MASS,
            base_stiffness=BASE_STIFFNESS,
            anchor_y=ANCHOR_POINT[1],
            sponge_start_frac=SPONGE_START_FRAC,
        )
        chain.apply_gaussian_pulse(PULSE_CENTER, PULSE_WIDTH, amplitude=0.6)

        # 2. Visual Setup
        ceiling = Line(LEFT * 2, RIGHT * 2, stroke_width=4).move_to(ANCHOR_POINT + UP * 0.1)

        # Create blocks (tiny squares). We color the sponge boundary red for visibility.
        blocks = VGroup(*[
            Square(
                side_length=BLOCK_SIDE,
                fill_color=BLUE if i < chain.sponge_start else interpolate_color(
                    BLUE, RED, (i - chain.sponge_start) / SPONGE_PAD
                ),
                fill_opacity=1,
                stroke_width=0,
            )
            for i in range(NUM_BLOCKS)
        ])
        for i in range(NUM_BLOCKS):
            blocks[i].move_to([0, chain.position[i], 0])

        spring_chain = VMobject(stroke_color=GRAY_C, stroke_width=1.5)
        spring_chain.set_points_as_corners([ANCHOR_POINT] + [blocks[i].get_center() for i in range(NUM_BLOCKS)])

        # 3. Motion Recorder Setup
        axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[-0.2, 0.2, 0.1],  # Tighter bounds magnify the curve
            x_length=6,
            y_length=4,
            axis_config={"include_numbers": True},
        ).to_edge(RIGHT, buff=0.5)

        axes_labels = axes.get_axis_labels(Text("t", font_size=20), Text("disp", font_size=20))

        trace_curve = VMobject(stroke_color=YELLOW, stroke_width=2)
        link_line = DashedLine(color=YELLOW, dash_length=0.1, stroke_opacity=0.5)

        history: list[tuple[float, float]] = []

        title = Text("100-Block Chain with Sponge Boundary & Trace", font_size=28).to_edge(UP, buff=0.2)

        # Zoom-out handoff: the close-up pair dissolves into the full chain at the same anchor.
        self.play(
            FadeOut(self.intro_group, self.intro_ceiling, self.intro_title),
            FadeIn(ceiling, blocks, spring_chain, title, axes, axes_labels, trace_curve, link_line),
            run_time=1.0,
        )

        # 4. The Physics-Driven Update Loop, running in slow motion
        def update_system(mob, dt):
            chain.step(dt * TIME_SCALE, dt_sim=SIM_TIME_STEP)

            for i in range(NUM_BLOCKS):
                blocks[i].move_to([0, chain.position[i], 0])

            chain_points = [ANCHOR_POINT] + [blocks[i].get_center() for i in range(NUM_BLOCKS)]
            spring_chain.set_points_as_corners(chain_points)

            disp0 = chain.displacement()[0]
            if chain.time <= 10.0:
                history.append((chain.time, disp0))

            if len(history) > 1:
                t_points = [axes.c2p(h[0], h[1]) for h in history]
                trace_curve.set_points_as_corners(t_points)
                link_line.put_start_and_end_on(blocks[0].get_right(), t_points[-1])

        # Attach the single master updater to the spring chain
        spring_chain.add_updater(update_system)

        # To capture 10 full virtual seconds at a time_scale of 0.06, we must wait 10 / 0.06 real seconds
        self.wait(10)

    def _play_intro(self) -> None:
        """Static 2-mass spring picture, held briefly, then zoomed into the chain's top."""
        intro_chain = MassSpringChain1D(
            n=2,
            total_length=2 * INTRO_REST_LENGTH,
            total_mass=2 * INTRO_MASS,
            base_stiffness=INTRO_SPRING_CONSTANT / 2,
            gravity=INTRO_GRAVITY,
            anchor_y=ANCHOR_POINT[1],
            base_damping=INTRO_DAMPING,
        )

        block1 = Square(side_length=0.8, fill_color=BLUE_D, fill_opacity=0.8, color=WHITE)
        block2 = Square(side_length=0.8, fill_color=RED_D, fill_opacity=0.8, color=WHITE)
        block1.move_to([ANCHOR_POINT[0], intro_chain.position[0], 0])
        block2.move_to([ANCHOR_POINT[0], intro_chain.position[1], 0])

        spring1 = VMobject(stroke_color=GRAY_B, stroke_width=3)
        spring1.set_points_as_corners(spring_zigzag_points(ANCHOR_POINT, block1.get_top()))
        spring2 = VMobject(stroke_color=GRAY_B, stroke_width=3)
        spring2.set_points_as_corners(spring_zigzag_points(block1.get_bottom(), block2.get_top()))

        ceiling = Line(LEFT * 2, RIGHT * 2, stroke_width=4).move_to(ANCHOR_POINT + UP * 0.1)
        title = Text("Two Coupled Masses", font_size=28).to_edge(UP, buff=0.2)

        self.add(ceiling, spring1, spring2, block1, block2, title)
        self.wait(INTRO_HOLD_TIME)

        # Zoom out: shrink the close-up pair down to the scale of the full chain's top.
        intro_group = VGroup(spring1, spring2, block1, block2)
        zoom_scale = BLOCK_SIDE / block1.width
        self.play(
            intro_group.animate.scale(zoom_scale).move_to(ANCHOR_POINT + DOWN * 0.1),
            run_time=ZOOM_OUT_RUN_TIME,
        )

        self.intro_group = intro_group
        self.intro_ceiling = ceiling
        self.intro_title = title