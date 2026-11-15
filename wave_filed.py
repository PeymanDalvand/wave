


"""2D acoustic wavefield propagation through a layered earth model.

The physics is driven by ``seismic_lib.AcousticWaveSolver2D``, a finite
difference solver built on the `findiff` FD-operator library, while this
module is only responsible for turning the simulation state into a Manim
animation (velocity-model image, Ricker source diagram, receiver traces,
and a schematic spring/ball texture over the deeper layers).
"""

import hashlib
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import av
import manim
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
SIMULATION_SECONDS = 1.5
WAVEFIELD_SAMPLE_STEPS = 5
BALL_STEP = 5
TRACE_WINDOW = 0.3  # seconds of receiver-trace history kept on screen

ANIMATION_TEXT = {
    "title": "Wave Propagation",
    "receiver_1": "R1",
    "receiver_2": "R2",
    "receiver_3": "R3",
    "receiver_4": "R4",
    "receiver_5": "R5",
    "receiver_6": "R6",
    "receiver_7": "R7",
    "receiver_8": "R8",
    "receiver_9": "R9",
    "receiver_10": "R10",
    "source": "Ricker source",
    "wavelet_title": "Ricker Wavelet (source)",
    "time_axis": "t (s)",
    "amplitude_axis": "amplitude",
    "trace_1": "R1 trace",
    "trace_2": "R2 trace",
}


def build_velocity_model() -> np.ndarray:
    """Three-layer velocity model: water over two increasingly stiff rock layers."""
    velocity = np.ones((GRID_NZ, GRID_NX)) * 2000.0
    velocity[LAYER_2_START:LAYER_3_START, :] = 4000.0
    velocity[LAYER_3_START:, :] = 5000.0
    return velocity


def load_wavefield_cache(cache_dir=None) -> tuple[np.ndarray, np.ndarray]:
    """Load or precompute physics-only snapshots, independent of scene styling and FPS."""
    velocity = build_velocity_model()
    steps = int(np.ceil(SIMULATION_SECONDS / TIME_STEP))
    snapshot_steps = np.minimum(
        np.arange(int(np.ceil(steps / WAVEFIELD_SAMPLE_STEPS)) + 1) * WAVEFIELD_SAMPLE_STEPS, steps
    )
    times = snapshot_steps * TIME_STEP
    parameters = dict(
        nz=GRID_NZ, nx=GRID_NX, dx=GRID_SPACING, dt=TIME_STEP, boundary_pad=BOUNDARY_PAD,
    )
    metadata = dict(
        format_version=1, **parameters, steps=steps, sample_steps=WAVEFIELD_SAMPLE_STEPS,
        source_row=SOURCE_ROW, source_col=SOURCE_COL,
        source_frequency=SOURCE_FREQUENCY, source_amplitude=SOURCE_AMPLITUDE,
    )
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("ascii"))
    digest.update(velocity.tobytes())
    digest.update(inspect.getsource(AcousticWaveSolver2D).encode("utf-8"))
    digest.update(inspect.getsource(ricker_wavelet).encode("utf-8"))
    directory = Path(cache_dir) if cache_dir is not None else Path(__file__).parent / "media" / "wavefield_cache"
    directory.mkdir(parents=True, exist_ok=True)
    cache_path = directory / f"{digest.hexdigest()}.npy"
    expected_shape = (len(times), GRID_NZ, GRID_NX)
    if cache_path.exists():
        try:
            frames = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if frames.shape == expected_shape and frames.dtype == np.float32:
                print(f"Reusing wavefield cache: {cache_path}")
                return frames, times
            del frames
        except (OSError, ValueError, EOFError):
            pass

    print(f"Computing {steps} wave steps; saving wavefield cache: {cache_path}")
    temporary_path = directory / f"{digest.hexdigest()}.{uuid4().hex}.tmp.npy"
    frames = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=np.float32, shape=expected_shape)
    try:
        solver = AcousticWaveSolver2D(velocity=velocity, **parameters)
        frames[0] = solver.u_curr
        for index, target_step in enumerate(snapshot_steps[1:], start=1):
            while solver.time_step < target_step:
                pulse = ricker_wavelet(solver.time_step * TIME_STEP, SOURCE_FREQUENCY) * SOURCE_AMPLITUDE
                solver.inject_source(SOURCE_ROW, SOURCE_COL, pulse)
                solver.advance()
            frames[index] = solver.u_curr
        frames.flush()
        frames = None
        temporary_path.replace(cache_path)
    finally:
        if temporary_path.exists():
            frames = None
            temporary_path.unlink()
    return np.load(cache_path, mmap_mode="r", allow_pickle=False), times


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


class WavefieldAnimation(Scene):
    """The reusable animated geometry, without editable text."""

    include_labels = False

    def construct(self):
        wavefields, sample_times = load_wavefield_cache()
        playback = ValueTracker(0.0)
        visuals, labels = self._build_layout()
        self.add(*visuals)
        if self.include_labels:
            self.add(*labels)
        img = self._image
        r1_row, r1_col = self._receiver_grids[0]
        r2_row, r2_col = self._receiver_grids[1]
        ricker_diagram = self._ricker_diagram
        source_overlay = self._source_overlay
        r1_panel, r2_panel = self._trace_panels

        layer_texture = LayerTexture(img, LAYER_2_START, LAYER_3_START, GRID_NZ, GRID_NX)
        self.add(layer_texture.springs, layer_texture.balls)

        r1_history = list(zip(sample_times, np.clip(wavefields[:, r1_row, r1_col] / 0.1, -1, 1)))
        r2_history = list(zip(sample_times, np.clip(wavefields[:, r2_row, r2_col] / 0.1, -1, 1)))

        def update_wavefield(mob, dt):
            index = int(np.clip(np.searchsorted(sample_times, playback.get_value(), side="right") - 1, 0, len(sample_times) - 1))
            normalized = np.nan_to_num(wavefields[index]) / 0.1
            amp = np.clip(normalized, -1, 1)
            mob.pixel_array = np.concatenate(
                [wavefield_to_rgb(amp), np.full((GRID_NZ, GRID_NX, 1), 255, dtype=np.uint8)], axis=2
            )

            layer_texture.update(normalized)

            t_now = sample_times[index]
            ricker_diagram.update(t_now)
            source_overlay.update(t_now)

            r1_panel.update(t_now, r1_history[:index + 1])
            r2_panel.update(t_now, r2_history[:index + 1])

        update_wavefield(img, 0.0)
        img.add_updater(update_wavefield)
        self.play(playback.animate.set_value(sample_times[-1]), run_time=SIM_SECONDS, rate_func=linear)
        img.remove_updater(update_wavefield)

    # -- construction helpers -------------------------------------------------

    def _build_layout(self):
        self._image = self._build_wavefield_image()
        self._receiver_grids = self._receiver_positions()
        self._receiver_markers, receiver_labels = self._build_receiver_array()
        self._ricker_diagram = RickerDiagram()
        self._source_overlay = SourceOverlay(
            self._ricker_diagram, self._grid_to_point(self._image, SOURCE_ROW, SOURCE_COL)
        )
        self._trace_panels = [
            TracePanel(ANIMATION_TEXT["trace_1"], self._ricker_diagram.axes),
        ]
        self._trace_panels.append(TracePanel(ANIMATION_TEXT["trace_2"], self._trace_panels[0].axes))
        diagram = self._ricker_diagram
        source = self._source_overlay
        visuals = [
            self._image, self._receiver_markers, diagram.axes, diagram.curve,
            diagram.tracker_dot, source.link_line, source.marker,
        ]
        labels = [
            Text(ANIMATION_TEXT["title"], font_size=32).to_edge(UP),
            receiver_labels, diagram.axis_labels, diagram.title, source.label,
        ]
        for axis in (diagram.axes.x_axis, diagram.axes.y_axis):
            labels.append(axis.numbers)
            axis.remove(axis.numbers)
        for panel in self._trace_panels:
            visuals.extend([panel.axes, panel.curve, panel.dot])
            labels.append(panel.label)
        return visuals, labels

    def _receiver_positions(self) -> list[tuple[int, int]]:
        image = self._build_wavefield_image()
        row, _ = self._point_to_grid(image, image.get_center() + np.array([-0.2, 1.5, 0]))
        origin_col = max(SOURCE_COL, BOUNDARY_PAD)
        last_col = GRID_NX - BOUNDARY_PAD - 1
        spacing = (last_col - origin_col) // 10
        if spacing < 1:
            raise ValueError("The model needs room for ten equally spaced receivers to the right of the source.")
        return [(row, origin_col + spacing * index) for index in range(1, 11)]

    def _build_receiver_array(self) -> tuple[VGroup, VGroup]:
        markers, labels = VGroup(), VGroup()
        spatial_order = sorted(range(10), key=lambda index: self._receiver_grids[index][1])
        ranks = {receiver_index: rank for rank, receiver_index in enumerate(spatial_order)}
        for receiver_index, (row, col) in enumerate(self._receiver_grids):
            point = self._grid_to_point(self._image, row, col)
            color = YELLOW if receiver_index < 2 else TEAL_A
            center = point + UP * (0.18 + 0.28 * (ranks[receiver_index] % 2))
            contact = Dot(point, radius=0.022, color=color)
            stem = Line(point, center + DOWN * 0.09, color=color, stroke_width=1, stroke_opacity=0.7)
            body = RoundedRectangle(
                width=0.13, height=0.18, corner_radius=0.035,
                stroke_color=color, stroke_width=1.2,
                fill_color="#152C32", fill_opacity=1,
            ).move_to(center)
            coil = VMobject(stroke_color=color, stroke_width=0.8)
            coil.set_points_as_corners(spring_zigzag_points(
                center + DOWN * 0.05, center + UP * 0.05, n_coils=3, width=0.016,
            ))
            highlight = Line(center + np.array([-0.04, 0.02, 0]),
                             center + np.array([-0.04, 0.055, 0]),
                             color=WHITE, stroke_width=1, stroke_opacity=0.6)
            markers.add(VGroup(contact, stem, body, coil, highlight))
            label = Text(ANIMATION_TEXT[f"receiver_{receiver_index + 1}"], font_size=14, color=color)
            label.move_to(center + UP * 0.2)
            labels.add(label)
        return markers, labels

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


class NumericalWavefield(WavefieldAnimation):
    """The wavefield animation with editable text overlays."""

    include_labels = True

    def construct(self):
        if config.transparent or config.save_last_frame or config.dry_run:
            super().construct()
            return
        wavefields, _ = load_wavefield_cache()
        video_path = load_rendered_wavefield_cache(wavefields)
        _, labels = self._build_layout()
        playback = ValueTracker(0.0)
        print("Overlaying current text on cached video; no lattice rendering or physics computation.", flush=True)
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            decoded_frames = container.decode(stream)
            frame = next(decoded_frames)
            frame_index = 0
            image = ImageMobject(frame.to_ndarray(format="rgba"))
            image.stretch_to_fit_width(config.frame_width)
            image.stretch_to_fit_height(config.frame_height)
            self.add(image, *labels)

            def update_video(mob, dt):
                nonlocal frame, frame_index
                target = min(round(playback.get_value() * float(stream.average_rate)), stream.frames - 1)
                if target <= frame_index:
                    return
                while frame_index < target:
                    frame = next(decoded_frames)
                    frame_index += 1
                mob.pixel_array = frame.to_ndarray(format="rgba")

            image.add_updater(update_video)
            try:
                with tempconfig({"disable_caching": True}):
                    self.play(playback.animate.set_value(SIM_SECONDS), run_time=SIM_SECONDS, rate_func=linear)
            finally:
                image.remove_updater(update_video)


class ReceiverTraces(WavefieldAnimation):
    """Ten receiver traces from the same cached wavefield, without the lattice."""

    def _trace_receiver_positions(self) -> list[tuple[int, int]]:
        return self._receiver_positions()

    def construct(self):
        wavefields, sample_times = load_wavefield_cache()
        self.receiver_positions = self._trace_receiver_positions()
        rows, columns = np.asarray(self.receiver_positions).T
        self.trace_values = np.asarray(wavefields[:, rows, columns])
        duration = float(sample_times[-1])
        if duration <= 0:
            raise ValueError("Receiver traces require a positive simulation duration.")

        title = Text("Receiver Traces", font_size=30).to_edge(UP)
        time_axis = NumberLine(
            x_range=[0, duration, duration / 3], length=config.frame_height - 2.6,
            include_numbers=True, include_tip=False, font_size=20,
            rotation=-PI / 2, label_direction=LEFT,
            decimal_number_config={"num_decimal_places": 2},
        ).move_to([-config.frame_width * 0.4, -0.25, 0])
        time_label = Text("Time (s)", font_size=20).rotate(PI / 2).next_to(time_axis, LEFT, buff=0.3)
        self.add(title, time_axis, time_label)

        left = time_axis.n2p(0)[0] + 0.7
        right = config.frame_width / 2 - 0.8
        top = time_axis.n2p(0)[1]
        bottom = time_axis.n2p(duration)[1]
        spacing = (right - left) / 9
        gain = spacing * 0.36 / max(float(np.max(np.abs(self.trace_values))), 1e-12)
        point_sets, curves, dots = [], [], []
        self.receiver_labels = VGroup()
        for receiver_index in range(10):
            baseline = left + receiver_index * spacing
            label = Text(f"R{receiver_index + 1}", font_size=20).move_to(
                np.array([baseline, top + 0.4, 0])
            )
            self.receiver_labels.add(label)
            self.add(Line([baseline, top, 0], [baseline, bottom, 0], stroke_width=0.7, stroke_opacity=0.25))
            points = np.column_stack([
                baseline + self.trace_values[:, receiver_index] * gain,
                top + sample_times / duration * (bottom - top),
                np.zeros(len(sample_times)),
            ])
            point_sets.append(points)
            color = YELLOW if receiver_index < 2 else TEAL_A
            curve = VMobject(stroke_color=color, stroke_width=1.8)
            curve.set_points_as_corners([points[0], points[0]])
            curves.append(curve)
            dots.append(Dot(points[0], radius=0.035, color=color))
        self.add(self.receiver_labels)

        playhead = Line([left - spacing * 0.4, top, 0], [right + spacing * 0.4, top, 0],
                        stroke_width=1, stroke_opacity=0.5)
        self.trace_curves = VGroup(*curves)
        plot = VGroup(self.trace_curves, *dots, playhead)
        self.add(plot)
        playback = ValueTracker(0.0)

        def update_traces(mob, dt):
            index = int(np.clip(np.searchsorted(sample_times, playback.get_value(), side="right") - 1,
                                0, len(sample_times) - 1))
            for points, curve, dot in zip(point_sets, curves, dots):
                visible = points[:index + 1] if index > 0 else np.repeat(points[:1], 2, axis=0)
                curve.set_points_as_corners(visible)
                dot.move_to(points[index])
            playhead.set_y(top + playback.get_value() / duration * (bottom - top))

        plot.add_updater(update_traces)
        try:
            self.play(playback.animate.set_value(duration), run_time=SIM_SECONDS, rate_func=linear)
        finally:
            plot.remove_updater(update_traces)


class LayerTexture:
    """Small "ball and spring" decoration bouncing with the local wave amplitude,
    overlaid on the deeper geological layers to suggest lattice bonding."""

    def __init__(self, img: ImageMobject, layer2_start: int, layer3_start: int, nz: int, nx: int):
        self.nz, self.nx = nz, nx
        ul = img.get_corner(UL)
        pixel_w, pixel_h = img.width / nx, img.height / nz
        self.ball_radius = 0.195 * BALL_STEP * min(pixel_w, pixel_h)
        self.spring_amplitude = self.ball_radius * 0.65
        self.ball_max_disp = BALL_STEP * min(pixel_w, pixel_h) * 0.3

        layer2 = self._layer_positions(ul, img, layer2_start, layer3_start)
        layer3 = self._layer_positions(ul, img, layer3_start, nz)

        self.layer2_balls, self.layer2_dots = self._make_balls(layer2, GRAY_B)
        self.layer3_balls, self.layer3_dots = self._make_balls(layer3, GRAY_C)
        self.layer2_springs, self.layer2_links = self._make_springs(layer2, GRAY_B, 4, 1.0)
        self.layer3_springs, self.layer3_links = self._make_springs(layer3, GRAY_C, 6, 0.7)

        self._layers = [
            (layer2, self.layer2_dots, self.layer2_links, GRAY_B, 4, 1.0),
            (layer3, self.layer3_dots, self.layer3_links, GRAY_C, 6, 0.7),
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
            body = Dot(
                point=pt, radius=self.ball_radius, color=interpolate_color(color, BLACK, 0.45),
                fill_opacity=0.9, stroke_width=0.5, stroke_color=color,
            )
            face = Dot(
                point=pt + UL * self.ball_radius * 0.12, radius=self.ball_radius * 0.78,
                color=color, fill_opacity=0.8, stroke_width=0,
            )
            highlight = Dot(
                point=pt + UL * self.ball_radius * 0.3, radius=self.ball_radius * 0.22,
                color=WHITE, fill_opacity=0.6, stroke_width=0,
            )
            mass = VGroup(body, face, highlight)
            dots[key] = mass
            balls.add(mass)
        return balls, dots

    def _spring_points(self, start: np.ndarray, end: np.ndarray, n_coils: int, width_scale: float) -> list:
        direction = end - start
        length = np.linalg.norm(direction)
        if length <= 2 * self.ball_radius:
            return [start, end]
        unit = direction / length
        start_tip = start + unit * self.ball_radius
        end_tip = end - unit * self.ball_radius
        lead = 0.12 * (length - 2 * self.ball_radius)
        coil = spring_zigzag_points(
            start_tip + unit * lead, end_tip - unit * lead,
            n_coils=n_coils, width=self.spring_amplitude * width_scale,
        )
        return [start_tip, *coil, end_tip]

    def _make_springs(self, positions: dict, color, n_coils: int, width_scale: float) -> tuple[VGroup, list]:
        springs, links = VGroup(), []
        for (row, col), pt in positions.items():
            for neighbor in ((row, col + BALL_STEP), (row + BALL_STEP, col)):
                if neighbor in positions:
                    spring = VMobject(stroke_color=color, stroke_width=0.85, stroke_opacity=0.35)
                    spring.set_points_as_corners(
                        self._spring_points(pt, positions[neighbor], n_coils, width_scale)
                    )
                    springs.add(spring)
                    links.append(((row, col), neighbor, spring))
        return springs, links

    def update(self, amp: np.ndarray) -> None:
        displacement = np.tanh(amp) * self.ball_max_disp
        for positions, dots, links, color, n_coils, width_scale in self._layers:
            displaced = {}
            for key in positions:
                row, col = key
                displaced[key] = positions[key] + np.array([0, displacement[row, col], 0])
                dots[key].move_to(displaced[key])
            for key1, key2, spring in links:
                spring.set_points_as_corners(
                    self._spring_points(displaced[key1], displaced[key2], n_coils, width_scale)
                )
                rest_length = np.linalg.norm(positions[key2] - positions[key1])
                strain = np.linalg.norm(displaced[key2] - displaced[key1]) / rest_length - 1.0
                strength = min(abs(strain) * 5, 1.0)
                spring.set_stroke(
                    color=interpolate_color(color, RED_C if strain > 0 else BLUE_B, strength * 0.8),
                    width=0.85 + strength * 0.5,
                    opacity=0.35 + strength * 0.4,
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

        self.axis_labels = self.axes.get_axis_labels(
            Text(ANIMATION_TEXT["time_axis"], font_size=20), Text(ANIMATION_TEXT["amplitude_axis"], font_size=20)
        )
        self.curve = self.axes.plot(
            lambda t: ricker_wavelet(t, SOURCE_FREQUENCY), x_range=[-0.05, 0.15], color=YELLOW, stroke_width=3
        )
        self.title = Text(ANIMATION_TEXT["wavelet_title"], font_size=22).next_to(self.axes, UP, buff=0.3)

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
        self.label = Text(ANIMATION_TEXT["source"], font_size=16, color=YELLOW).next_to(self.marker, UP, buff=0.15)
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


def rendered_wavefield_cache_path(wavefields: np.ndarray, cache_dir=None) -> Path:
    """Identify a rendered background without including editable caption values."""
    metadata = dict(
        format_version=1,
        physics_cache=Path(wavefields.filename).stem,
        manim_version=manim.__version__,
        renderer=str(config.renderer),
        pixel_width=config.pixel_width, pixel_height=config.pixel_height,
        frame_width=config.frame_width, frame_height=config.frame_height,
        frame_rate=float(config.frame_rate), background_color=str(config.background_color),
        background_opacity=config.background_opacity,
        animation_seconds=SIM_SECONDS, ball_step=BALL_STEP, trace_window=TRACE_WINDOW,
        layer_2_start=LAYER_2_START, layer_3_start=LAYER_3_START,
    )
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("ascii"))
    for implementation in (
        WavefieldAnimation, LayerTexture, RickerDiagram, SourceOverlay, TracePanel,
        wavefield_to_rgb, spring_zigzag_points,
    ):
        digest.update(inspect.getsource(implementation).encode("utf-8"))
    directory = Path(cache_dir) if cache_dir is not None else Path(__file__).parent / "media" / "wavefield_video_cache"
    return directory / f"{digest.hexdigest()}.mp4"


def _rendered_wavefield_is_valid(path: Path) -> bool:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            return (
                stream.width == config.pixel_width and stream.height == config.pixel_height
                and stream.frames == int(np.ceil(SIM_SECONDS * float(config.frame_rate)))
                and stream.average_rate is not None
                and np.isclose(float(stream.average_rate), float(config.frame_rate))
            )
    except (OSError, ValueError, IndexError, av.error.FFmpegError):
        return False


def load_rendered_wavefield_cache(wavefields: np.ndarray, cache_dir=None) -> Path:
    """Render the text-free animation once, then reuse the encoded video."""
    cache_path = rendered_wavefield_cache_path(wavefields, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and _rendered_wavefield_is_valid(cache_path):
        print(f"Rendered video cache HIT: {cache_path}", flush=True)
        return cache_path

    print(f"Rendered video cache MISS: rendering the lattice once into {cache_path}", flush=True)
    with TemporaryDirectory(prefix="render-", dir=cache_path.parent) as directory:
        temporary_path = Path(directory) / "animation.mp4"
        with tempconfig({
            "output_file": str(temporary_path),
            "partial_movie_dir": str(Path(directory) / "partials"),
            "preview": False, "show_in_file_browser": False,
            "write_to_movie": True, "save_last_frame": False, "write_all": False,
            "format": "mp4", "transparent": False, "dry_run": False,
            "disable_caching": True, "save_sections": False,
            "from_animation_number": 0, "upto_animation_number": float("inf"),
        }):
            WavefieldAnimation().render()
        if not _rendered_wavefield_is_valid(temporary_path):
            raise RuntimeError("The rendered wavefield video is incomplete; no cache was published.")
        temporary_path.replace(cache_path)
    return cache_path