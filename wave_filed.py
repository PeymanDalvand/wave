"""2D acoustic wavefield propagation through a layered earth model.

The physics is driven by ``seismic_lib.AcousticWaveSolver2D``, a finite
difference solver built on the `findiff` FD-operator library, while this
module is only responsible for turning the simulation state into a Manim
animation (velocity-model image, Ricker source diagram, receiver traces,
and a schematic spring/ball texture over the deeper layers).
"""

from manim import *
import numpy as np
from PIL import Image

from seismic_lib import AcousticWaveSolver2D, ricker_wavelet, spring_zigzag_points

# Grid & physical parameters
GRID_NZ, GRID_NX = 200, 200
GRID_SPACING = 10.0
TIME_STEP = 0.001
SOURCE_ROW, SOURCE_COL = GRID_NZ // 4, GRID_NX // 4
SOURCE_FREQUENCY = 15.0
SOURCE_AMPLITUDE = 0.5
BOUNDARY_PAD = 25

# Layer boundaries (row indices) and their P-wave velocities
LAYER_2_START = GRID_NZ // 2
LAYER_3_START = 3 * GRID_NZ // 4

# Animation pacing
SPEED_FACTOR = 1 / 3
SIM_SECONDS = 20 / SPEED_FACTOR
BALL_STEP = 10
TRACE_WINDOW = 0.3  # seconds of receiver-trace history kept on screen


def build_velocity_model() -> np.ndarray:
    """Three-layer velocity model: water over two increasingly stiff rock layers."""
    velocity = np.ones((GRID_NZ, GRID_NX)) * 2000.0
    velocity[LAYER_2_START:LAYER_3_START, :] = 4000.0
    velocity[LAYER_3_START:, :] = 5000.0
    return velocity


def wavefield_to_rgb(amplitude: np.ndarray) -> np.ndarray:
    """Colorize the normalized wavefield: blue water tint, green rock layers,
    with red/blue overlaid for positive/negative wave amplitude."""
    rgb = np.zeros((GRID_NZ, GRID_NX, 3), dtype=np.uint8)

    rgb[:LAYER_2_START, :, 0] = 15
    rgb[:LAYER_2_START, :, 1] = 60
    rgb[:LAYER_2_START, :, 2] = 110

    positive = (amplitude > 0) * amplitude * 255
    negative = (amplitude < 0) * (-amplitude) * 255
    rgb[..., 0] = np.clip(rgb[..., 0].astype(np.float64) + positive, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(rgb[..., 2].astype(np.float64) + negative, 0, 255).astype(np.uint8)

    rgb[LAYER_2_START:LAYER_3_START, :, 1] = 80
    rgb[LAYER_3_START:, :, 1] = 140
    return rgb


class NumericalWavefield(Scene):
    """Animates a Ricker-source acoustic wavefield spreading through a layered model."""

    def construct(self):
        solver = AcousticWaveSolver2D(
            nz=GRID_NZ,
            nx=GRID_NX,
            dx=GRID_SPACING,
            dt=TIME_STEP,
            velocity=build_velocity_model(),
            boundary_pad=BOUNDARY_PAD,
        )

        img = self._build_wavefield_image()
        title = Text("Wave Propagation with Absorbing Boundaries", font_size=32).to_edge(UP)
        self.add(title, img)

        receiver1, r1_row, r1_col = self._build_receiver(img, [-0.2, 1.5, 0], "R1")
        receiver2, r2_row, r2_col = self._build_receiver(img, [1.3, 1.5, 0], "R2")
        self.add(receiver1, receiver2)

        layer_texture = LayerTexture(img, LAYER_2_START, LAYER_3_START, GRID_NZ, GRID_NX)
        self.add(layer_texture.springs, layer_texture.balls)

        ricker_diagram = RickerDiagram()
        self.add(*ricker_diagram.mobjects)

        source_point = self._grid_to_point(img, SOURCE_ROW, SOURCE_COL)
        source_overlay = SourceOverlay(ricker_diagram, source_point)
        self.add(*source_overlay.mobjects)

        r1_panel = TracePanel("R1 trace", ricker_diagram.axes)
        r2_panel = TracePanel("R2 trace", r1_panel.axes)
        self.add(*r1_panel.mobjects, *r2_panel.mobjects)

        step_accumulator = 0.0
        r1_history: list[tuple[float, float]] = []
        r2_history: list[tuple[float, float]] = []

        def update_wavefield(mob, dt):
            nonlocal step_accumulator

            step_accumulator += 5 * SPEED_FACTOR
            steps_per_frame = int(step_accumulator)
            step_accumulator -= steps_per_frame

            for _ in range(steps_per_frame):
                t_now = solver.time_step * TIME_STEP
                solver.inject_source(SOURCE_ROW, SOURCE_COL, ricker_wavelet(t_now, SOURCE_FREQUENCY) * SOURCE_AMPLITUDE)
                solver.advance()

            amp = np.clip(np.nan_to_num(solver.u_curr) / 0.1, -1, 1)
            mob.pixel_array = np.concatenate(
                [wavefield_to_rgb(amp), np.full((GRID_NZ, GRID_NX, 1), 255, dtype=np.uint8)], axis=2
            )

            layer_texture.update(amp)

            t_now = solver.time_step * TIME_STEP
            ricker_diagram.update(t_now)
            source_overlay.update(t_now)

            r1_history.append((t_now, amp[r1_row, r1_col]))
            r2_history.append((t_now, amp[r2_row, r2_col]))
            r1_panel.update(t_now, r1_history)
            r2_panel.update(t_now, r2_history)

        img.add_updater(update_wavefield)
        self.wait(SIM_SECONDS)
        img.remove_updater(update_wavefield)

    # -- construction helpers -------------------------------------------------

    def _build_wavefield_image(self) -> ImageMobject:
        img = ImageMobject(np.zeros((GRID_NZ, GRID_NX, 3), dtype=np.uint8))
        img.set_resampling_algorithm(Image.NEAREST)
        img.scale(3.5)
        img.to_edge(RIGHT, buff=0.8)
        return img

    def _build_receiver(self, img: ImageMobject, offset, label_text: str):
        marker = Triangle(color=YELLOW, fill_opacity=1, stroke_width=1).scale(0.15).rotate(PI)
        marker.move_to(img.get_center() + np.array(offset))
        label = Text(label_text, font_size=20, color=YELLOW).next_to(marker, UP, buff=0.05)
        row, col = self._point_to_grid(img, marker.get_center())
        return VGroup(marker, label), row, col

    def _point_to_grid(self, img: ImageMobject, point: np.ndarray) -> tuple[int, int]:
        ul = img.get_corner(UL)
        col = (point[0] - ul[0]) / img.width * GRID_NX - 0.5
        row = -(point[1] - ul[1]) / img.height * GRID_NZ - 0.5
        return int(np.clip(round(row), 0, GRID_NZ - 1)), int(np.clip(round(col), 0, GRID_NX - 1))

    def _grid_to_point(self, img: ImageMobject, row: int, col: int) -> np.ndarray:
        ul = img.get_corner(UL)
        return ul + np.array([(col + 0.5) / GRID_NX * img.width, -(row + 0.5) / GRID_NZ * img.height, 0])


class LayerTexture:
    """Small "ball and spring" decoration bouncing with the local wave amplitude,
    overlaid on the deeper geological layers to suggest lattice bonding."""

    def __init__(self, img: ImageMobject, layer2_start: int, layer3_start: int, nz: int, nx: int):
        self.nz, self.nx = nz, nx
        ul = img.get_corner(UL)
        pixel_w, pixel_h = img.width / nx, img.height / nz
        self.ball_radius = 0.3 * BALL_STEP * min(pixel_w, pixel_h)
        self.spring_amplitude = self.ball_radius * 0.8
        self.ball_max_disp = BALL_STEP * min(pixel_w, pixel_h) * 0.4

        layer2 = self._layer_positions(ul, img, layer2_start, layer3_start)
        layer3 = self._layer_positions(ul, img, layer3_start, nz)

        self.layer2_balls, self.layer2_dots = self._make_balls(layer2, GRAY_A)
        self.layer3_balls, self.layer3_dots = self._make_balls(layer3, GRAY_D)
        self.layer2_springs, self.layer2_links = self._make_springs(layer2, GRAY_A)
        self.layer3_springs, self.layer3_links = self._make_springs(layer3, GRAY_D)

        self._layers = [
            (layer2, self.layer2_dots, self.layer2_links),
            (layer3, self.layer3_dots, self.layer3_links),
        ]

    @property
    def balls(self) -> VGroup:
        return VGroup(self.layer2_balls, self.layer3_balls)

    @property
    def springs(self) -> VGroup:
        return VGroup(self.layer2_springs, self.layer3_springs)

    def _layer_positions(self, ul, img: ImageMobject, row_start: int, row_end: int) -> dict:
        positions = {}
        for row in range(row_start, row_end, BALL_STEP):
            for col in range(0, self.nx, BALL_STEP):
                positions[(row, col)] = ul + np.array(
                    [(col + 0.5) / self.nx * img.width, -(row + 0.5) / self.nz * img.height, 0]
                )
        return positions

    def _make_balls(self, positions: dict, color) -> tuple[VGroup, dict]:
        balls, dots = VGroup(), {}
        for key, pt in positions.items():
            dot = Dot(point=pt, radius=self.ball_radius, color=color, fill_opacity=0.55, stroke_width=0)
            dots[key] = dot
            balls.add(dot)
        return balls, dots

    def _make_springs(self, positions: dict, color) -> tuple[VGroup, list]:
        springs, links = VGroup(), []
        for (row, col), pt in positions.items():
            for neighbor in ((row, col + BALL_STEP), (row + BALL_STEP, col)):
                if neighbor in positions:
                    spring = VMobject(stroke_color=color, stroke_width=1.5, stroke_opacity=0.6)
                    spring.set_points_as_corners(
                        spring_zigzag_points(pt, positions[neighbor], n_coils=4, width=self.spring_amplitude)
                    )
                    springs.add(spring)
                    links.append(((row, col), neighbor, spring))
        return springs, links

    def update(self, amp: np.ndarray) -> None:
        for positions, dots, links in self._layers:
            displaced = {}
            for key in positions:
                row, col = key
                displaced[key] = positions[key] + np.array([0, amp[row, col] * self.ball_max_disp, 0])
                dots[key].move_to(displaced[key])
            for key1, key2, spring in links:
                spring.set_points_as_corners(
                    spring_zigzag_points(displaced[key1], displaced[key2], n_coils=4, width=self.spring_amplitude)
                )


class RickerDiagram:
    """A scrolling oscilloscope-style plot of the injected Ricker source wavelet."""

    def __init__(self):
        self.axes = Axes(
            x_range=[-0.05, 0.15, 0.05],
            y_range=[-0.5, 1.1, 0.3],
            x_length=4.6,
            y_length=3.2,
            axis_config={"include_numbers": True, "font_size": 16, "stroke_width": 2},
            tips=False,
        ).to_edge(LEFT, buff=0.7)

        self.axis_labels = self.axes.get_axis_labels(Text("t (s)", font_size=20), Text("amplitude", font_size=20))
        self.curve = self.axes.plot(
            lambda t: ricker_wavelet(t, SOURCE_FREQUENCY), x_range=[-0.05, 0.15], color=YELLOW, stroke_width=3
        )
        self.title = Text("Ricker Wavelet (source)", font_size=22).next_to(self.axes, UP, buff=0.3)

        self.dot_x = self.axes.c2p(0, 0)[0]
        self.tracker_dot = Dot(radius=0.16, color=WHITE).move_to(self.axes.c2p(0, ricker_wavelet(0, SOURCE_FREQUENCY)))

    @property
    def mobjects(self):
        return [self.axes, self.axis_labels, self.curve, self.title, self.tracker_dot]

    def update(self, t_now: float) -> None:
        current_val = ricker_wavelet(t_now, SOURCE_FREQUENCY)
        self.curve.become(
            self.axes.plot(
                lambda rel_t: ricker_wavelet(rel_t + t_now, SOURCE_FREQUENCY),
                x_range=[-0.05, 0.15], color=YELLOW, stroke_width=3,
            )
        )
        y_now = self.axes.c2p(0, current_val)[1]
        self.tracker_dot.move_to([self.dot_x, y_now, 0]).set_color(RED if current_val > 0 else BLUE)


class SourceOverlay:
    """Pulsing marker at the wavefield injection point, linked back to the Ricker diagram."""

    def __init__(self, diagram: RickerDiagram, source_point: np.ndarray):
        self.source_point = source_point
        self.marker = Circle(radius=0.08, color=WHITE, fill_opacity=0.9).move_to(source_point)
        self.label = Text("Ricker source", font_size=16, color=YELLOW).next_to(self.marker, UP, buff=0.15)
        self.link_line = DashedLine(diagram.axes.get_right(), source_point, color=YELLOW, stroke_width=1.5, dash_length=0.08)

    @property
    def mobjects(self):
        return [self.link_line, self.label, self.marker]

    def update(self, t_now: float) -> None:
        current_val = ricker_wavelet(t_now, SOURCE_FREQUENCY)
        color = RED if current_val > 0 else BLUE
        radius = 0.08 + min(abs(current_val), 1.0) * 0.18
        self.marker.become(Circle(radius=radius, color=color, fill_opacity=0.9).move_to(self.source_point))


class TracePanel:
    """Scrolling wiggle-trace readout of the wavefield amplitude at a receiver."""

    def __init__(self, label_text: str, anchor: Axes):
        self.axes = Axes(
            x_range=[-TRACE_WINDOW, 0, TRACE_WINDOW / 3],
            y_range=[-1.1, 1.1, 0.5],
            x_length=4.6,
            y_length=0.9,
            axis_config={"include_numbers": False, "stroke_width": 1.5},
            tips=False,
        ).next_to(anchor, DOWN, buff=0.3)
        self.label = Text(label_text, font_size=16, color=YELLOW).move_to(
            self.axes.get_corner(UL) + RIGHT * 0.45 + DOWN * 0.15
        )
        self.curve = VMobject(stroke_color=YELLOW, stroke_width=2)
        self.dot = Dot(radius=0.06, color=YELLOW).move_to(self.axes.c2p(0, 0))

    @property
    def mobjects(self):
        return [self.axes, self.label, self.curve, self.dot]

    def update(self, t_now: float, history: list[tuple[float, float]]) -> None:
        recent = [(t - t_now, v) for t, v in history if t_now - t <= TRACE_WINDOW]
        if len(recent) >= 2:
            self.curve.set_points_as_corners([self.axes.c2p(rt, v) for rt, v in recent])
        last_val = recent[-1][1] if recent else 0.0
        self.dot.move_to(self.axes.c2p(0, last_val)).set_color(RED if last_val > 0 else BLUE)