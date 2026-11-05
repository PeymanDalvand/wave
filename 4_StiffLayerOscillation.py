"""A 130-block chain with a stiffer middle layer and a sponge boundary.

Shows how a local stiffness contrast (representing a higher-velocity rock
layer) partially reflects a Gaussian pulse, while the far end absorbs the
transmitted energy instead of reflecting it back. Physics lives in
``seismic_lib.MassSpringChain1D``.
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
TIME_SCALE = 0.03
ANCHOR_POINT = np.array([0, 3.5, 0])


class StiffLayerOscillation(Scene):
    """N-block chain: normal layer -> stiff layer -> sponge boundary, with a live trace."""

    def construct(self):
        # 1. Physics
        chain = MassSpringChain1D(
            n=NUM_BLOCKS,
            total_length=TOTAL_LENGTH,
            total_mass=TOTAL_MASS,
            base_stiffness=BASE_STIFFNESS,
            anchor_y=ANCHOR_POINT[1],
            stiff_band=STIFF_BAND,
            stiff_multiplier=STIFF_MULTIPLIER,
            sponge_start_frac=SPONGE_START_FRAC,
            sponge_max_damping=SPONGE_MAX_DAMPING,
        )
        chain.apply_gaussian_pulse(PULSE_CENTER, PULSE_WIDTH, amplitude=0.6)
        stiff_start = int(NUM_BLOCKS * STIFF_BAND[0])
        sponge_start = chain.sponge_start
        sponge_pad = NUM_BLOCKS - sponge_start

        # 2. Visual Setup
        ceiling = Line(LEFT * 2, RIGHT * 2, stroke_width=4).move_to(ANCHOR_POINT + UP * 0.1)

        def block_color(i):
            if i < stiff_start:
                return BLUE
            elif i < sponge_start:
                return GREEN
            return interpolate_color(GREEN, RED, (i - sponge_start) / sponge_pad)

        block_size = min(0.3, 4.0 / NUM_BLOCKS)
        spring_thickness = max(1.0, 150.0 / NUM_BLOCKS)

        blocks = VGroup(*[
            Square(side_length=block_size, fill_color=block_color(i), fill_opacity=1, stroke_width=0)
            for i in range(NUM_BLOCKS)
        ])

        spring_chain = VMobject(stroke_color=GRAY_C, stroke_width=spring_thickness)

        title = Text(f"{NUM_BLOCKS}-Block Chain: Stiff Middle Layer & Sponge Bottom", font_size=28).to_edge(UP, buff=0.2)

        # Moved labels to the LEFT so they don't overlap with the new axes on the right
        top_label = Text("Normal Layer", font_size=16, color=BLUE).next_to(blocks[int(NUM_BLOCKS * 0.2)], LEFT, buff=1.0)
        stiff_label = Text("Stiff Layer", font_size=16, color=GREEN).next_to(blocks[int(NUM_BLOCKS * 0.58)], LEFT, buff=1.0)
        sponge_label = Text("Sponge Boundary", font_size=16, color=RED).next_to(blocks[int(NUM_BLOCKS * 0.88)], LEFT, buff=1.0)

        # 3. Motion Recorder Setup
        axes = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-0.2, 0.2, 0.1],  # Tight bounds to naturally magnify the wave pulse
            x_length=5,
            y_length=4,
            axis_config={"include_numbers": True},
        ).to_edge(RIGHT, buff=0.5)

        axes_labels = axes.get_axis_labels(Text("t", font_size=20), Text("disp", font_size=20))

        trace_curve = VMobject(stroke_color=YELLOW, stroke_width=2)
        link_line = DashedLine(color=YELLOW, dash_length=0.1, stroke_opacity=0.5)

        history: list[tuple[float, float]] = []

        self.add(
            ceiling, spring_chain, blocks, title,
            top_label, stiff_label, sponge_label,
            axes, axes_labels, trace_curve, link_line,
        )

        # 4. The Physics-Driven Update Loop
        def update_system(mob, dt):
            chain.step(dt * TIME_SCALE, dt_sim=SIM_TIME_STEP)

            for i in range(NUM_BLOCKS):
                blocks[i].move_to([0, chain.position[i], 0])

            chain_points = [ANCHOR_POINT] + [blocks[i].get_center() for i in range(NUM_BLOCKS)]
            spring_chain.set_points_as_corners(chain_points)

            disp0 = chain.displacement()[0]
            if chain.time <= 15.0:
                history.append((chain.time, disp0))

            if len(history) > 1:
                t_points = [axes.c2p(h[0], h[1]) for h in history]
                trace_curve.set_points_as_corners(t_points)
                link_line.put_start_and_end_on(blocks[0].get_right(), t_points[-1])

        spring_chain.add_updater(update_system)

        # Adjusted the wait time mathematically so 15 full virtual seconds execute
        self.wait(30)