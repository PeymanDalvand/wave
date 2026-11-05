"""A 2D mass-spring lattice modeling a seismic shot gather.

A vertical Gaussian pulse injected at the surface propagates through a
stiffer middle layer and a sponge-absorbed lower boundary, while ten
"surface receivers" record scrolling wiggle traces -- the discrete,
lumped-parameter analogue of a shot gather. Physics lives in
``seismic_lib.MassSpringGrid2D``.
"""

from manim import *
import numpy as np

from seismic_lib import MassSpringGrid2D

N_ROWS = 50
N_COLS = 101
TOTAL_LENGTH = 6.0
TOTAL_MASS = 5.0
BASE_STIFFNESS_Y = 30.0
BASE_STIFFNESS_X = 5.0
STIFF_BAND = (0.40, 0.76)
STIFF_MULTIPLIER = 8.0
SPONGE_START_FRAC = 0.76
SPONGE_MAX_DAMPING = 25.0
SPACING_X = 0.12
ANCHOR_Y = 3.5
PULSE_ROW = 5
PULSE_ROW_WIDTH = N_ROWS * 0.05
PULSE_COL_WIDTH = N_COLS * 0.05
SIM_TIME_STEP = 0.002
TIME_SCALE = 0.04
RECEIVER_COLUMNS = [N_COLS // 2 + 5 * i for i in range(1, 11)]


class Grid2DOscillation(Scene):
    """A 2D lattice shot-gather simulation with ten scrolling receiver traces."""

    def construct(self):
        # 1. Physics
        grid = MassSpringGrid2D(
            n_rows=N_ROWS,
            n_cols=N_COLS,
            total_length=TOTAL_LENGTH,
            total_mass=TOTAL_MASS,
            base_stiffness_y=BASE_STIFFNESS_Y,
            base_stiffness_x=BASE_STIFFNESS_X,
            anchor_y=ANCHOR_Y,
            stiff_band=STIFF_BAND,
            stiff_multiplier=STIFF_MULTIPLIER,
            sponge_start_frac=SPONGE_START_FRAC,
            sponge_max_damping=SPONGE_MAX_DAMPING,
        )
        center_col = N_COLS // 2
        grid.apply_gaussian_pulse(PULSE_ROW, center_col, PULSE_ROW_WIDTH, PULSE_COL_WIDTH, amplitude=1.0)

        stiff_start = int(N_ROWS * STIFF_BAND[0])
        sponge_start = grid.sponge_start
        sponge_pad = N_ROWS - sponge_start

        # 2. Spatial layout
        start_x = -(N_COLS // 2) * SPACING_X
        x_positions = np.linspace(start_x, -start_x, N_COLS)

        # 3. Visual Setup (Explicit Blocks and Springs)
        def block_color(i):
            if i < stiff_start:
                return BLUE
            elif i < sponge_start:
                return GREEN
            return interpolate_color(GREEN, RED, (i - sponge_start) / sponge_pad)

        block_size = 0.06
        spring_thickness = 1.0

        blocks = VGroup(*[
            Square(side_length=block_size, fill_color=block_color(i), fill_opacity=1, stroke_width=0)
            .move_to([x_positions[j], grid.position[i, j], 0])
            for i in range(N_ROWS) for j in range(N_COLS)
        ])

        v_springs = VGroup(*[VMobject(stroke_color=GRAY_C, stroke_width=spring_thickness) for _ in range(N_COLS)])
        h_springs = VGroup(*[
            VMobject(stroke_color=GRAY_C, stroke_width=spring_thickness, stroke_opacity=0.5) for _ in range(N_ROWS)
        ])

        # Ten surface receivers to the right of the source, each with a scrolling trace
        receivers = VGroup(*[
            Triangle(fill_color=YELLOW, fill_opacity=1, stroke_width=0)
            .scale(0.08)
            .rotate(PI)
            .move_to([x_positions[col], ANCHOR_Y + 0.2, 0])
            for col in RECEIVER_COLUMNS
        ])

        traces = VGroup(*[VMobject(stroke_color=YELLOW, stroke_width=1.5) for _ in RECEIVER_COLUMNS])
        trace_history = {col: [] for col in RECEIVER_COLUMNS}

        title = Text("Seismic Shot Gather (10 Surface Receivers)", font_size=28).to_edge(UP, buff=0.2)
        self.add(v_springs, h_springs, blocks, receivers, traces, title)

        # 4. The Physics-Driven Update Loop
        def update_system(mob, dt):
            grid.step(dt * TIME_SCALE, dt_sim=SIM_TIME_STEP)

            for i in range(N_ROWS):
                for j in range(N_COLS):
                    blocks[i * N_COLS + j].move_to([x_positions[j], grid.position[i, j], 0])

            for j in range(N_COLS):
                chain_pts = [np.array([x_positions[j], ANCHOR_Y, 0])] + [
                    blocks[i * N_COLS + j].get_center() for i in range(N_ROWS)
                ]
                v_springs[j].set_points_as_corners(chain_pts)

            for i in range(N_ROWS):
                row_pts = [blocks[i * N_COLS + j].get_center() for j in range(N_COLS)]
                h_springs[i].set_points_as_corners(row_pts)

            # Update the traces scrolling downward with time
            displacement = grid.displacement()
            for idx, col in enumerate(RECEIVER_COLUMNS):
                disp = displacement[0, col]
                px = x_positions[col] + disp * 1.5
                py = ANCHOR_Y - (grid.time * 10.0)
                trace_history[col].append(np.array([px, py, 0]))
                if len(trace_history[col]) > 1:
                    traces[idx].set_points_as_corners(trace_history[col])

        blocks.add_updater(update_system)
        self.wait(15)