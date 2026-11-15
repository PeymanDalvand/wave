"""Presentation sequence: recordings, acoustic physics, source, and velocity.

Recorded data come from the existing numerical wavefield cache. Analytical
illustrations are labeled as such; these scenes do not perform an inversion.
"""

import importlib

from manim import *
import numpy as np

import wave_filed as wave
from seismic_lib import hyperbolic_traveltime, ricker_wavelet


SpringOscillationScene = importlib.import_module("1_SpringOscillation").SpringOscillation
CoupledOscillationScene = importlib.import_module("2_CoupledOscillation").CoupledOscillation
LargeCoupledOscillationScene = importlib.import_module("3_LargeCoupledOscillation").LargeCoupledOscillation
StiffLayerOscillationScene = importlib.import_module("4_StiffLayerOscillation").StiffLayerOscillation
Grid2DOscillationScene = importlib.import_module("6_Grid2DOscillation").Grid2DOscillation


PRESENTATION_PACE = 1.0
SOURCE_COLOR = YELLOW
MODEL_COLOR = TEAL_A


def _text(content, size=24, color=WHITE, width=12.0):
    label = Text(content, font_size=size, color=color)
    if label.width > width:
        label.scale_to_fit_width(width)
    return label


def _beat(scene, *animations, duration=1.0):
    scene.play(*animations, run_time=max(1 / config.frame_rate, duration * PRESENTATION_PACE))


def _hold(scene, seconds=1.5):
    scene.wait(max(1 / config.frame_rate, seconds * PRESENTATION_PACE))


def _heading(scene, number, title):
    step = _text(f"{number:02d} / 06", 16, MODEL_COLOR).to_corner(UL, buff=0.4)
    heading = _text(title, 32, width=10.5).move_to([0, 3.2, 0])
    rule = Line([-6.2, 2.8, 0], [6.2, 2.8, 0], stroke_width=1, stroke_opacity=0.3)
    _beat(scene, FadeIn(step), FadeIn(heading), Create(rule))


def _recorded_gather():
    frames, times = wave.load_wavefield_cache()
    positions = wave.ReceiverTraces()._trace_receiver_positions()
    rows, columns = np.asarray(positions).T
    amplitudes = np.asarray(frames[:, rows, columns])
    scale = 0.3 / max(float(np.max(np.abs(amplitudes))), 1e-12)
    curves, labels, baselines = VGroup(), VGroup(), VGroup()
    for receiver_index in range(10):
        center = -4.8 + receiver_index * (9.6 / 9)
        points = np.column_stack([
            center + amplitudes[:, receiver_index] * scale,
            1.75 - times / times[-1] * 3.7,
            np.zeros(len(times)),
        ])
        color = SOURCE_COLOR if receiver_index < 2 else MODEL_COLOR
        curve = VMobject(stroke_color=color, stroke_width=1.8)
        curve.set_points_as_corners(points)
        curves.add(curve)
        labels.add(_text(f"R{receiver_index + 1}", 18, color).move_to([center, 2.15, 0]))
        baselines.add(Line([center, 1.75, 0], [center, -1.95, 0],
                           stroke_width=0.6, stroke_opacity=0.2))
    axis = NumberLine(
        x_range=[0, float(times[-1]), float(times[-1]) / 3], length=3.7,
        rotation=-PI / 2, include_numbers=True, include_tip=False,
        label_direction=LEFT, font_size=16,
        decimal_number_config={"num_decimal_places": 2},
    )
    axis.shift(np.array([-5.55, 1.75, 0]) - axis.n2p(0))
    time_label = _text("Time (s)", 17).rotate(PI / 2).next_to(axis, LEFT, buff=0.2)
    return curves, labels, VGroup(baselines, axis, time_label)


class RecordedTraces(Scene):
    """Begin with the measurements, not with an already-known earth model."""

    def construct(self):
        _heading(self, 1, "Ten recordings. A hidden subsurface.")
        traces, labels, frame = _recorded_gather()
        _beat(self, FadeIn(frame), FadeIn(labels))
        _beat(self, LaggedStart(*(Create(trace) for trace in traces), lag_ratio=0.08), duration=3)
        question = _text("What subsurface could have produced these signals?", 28).move_to([0, -2.65, 0])
        caveat = _text("The traces constrain an earth model; they do not reveal it uniquely.", 19, GRAY_B)
        caveat.move_to([0, -3.25, 0])
        _beat(self, FadeIn(question))
        _hold(self, 2)
        _beat(self, FadeIn(caveat))
        _hold(self, 3)


class AcousticAssumptions(Scene):
    """Show longitudinal motion and the assumptions behind the scalar model."""

    def construct(self):
        _heading(self, 2, "How does a wave move through the earth?")
        particle_title = _text("Compression and expansion", 23, MODEL_COLOR).move_to([-3.3, 2.15, 0])
        direction = Arrow([-5.7, 1.4, 0], [-1.0, 1.4, 0], color=MODEL_COLOR, buff=0)
        direction_label = _text("Propagation", 18, MODEL_COLOR).next_to(direction, UP, buff=0.1)
        particles = VGroup()
        rest_positions = []
        for row in range(4):
            for col in range(18):
                position = np.array([-6.0 + col * 0.31, 0.65 - row * 0.45, 0])
                particles.add(Dot(position, radius=0.055, color=WHITE))
                rest_positions.append(position)
        motion_label = _text("Particle motion is parallel to propagation.", 19, GRAY_B, width=6)
        motion_label.move_to([-3.3, -1.35, 0])
        _beat(self, FadeIn(particle_title), GrowArrow(direction), FadeIn(direction_label),
              FadeIn(particles), FadeIn(motion_label))

        assumptions = VGroup(*[
            _text(content, 23, width=5.8) for content in (
                "Small pressure disturbances",
                "P waves only; no shear rigidity",
                "Constant density; variable wave speed",
                "Isotropic and lossless",
                "A 2D approximation to a 3D earth",
            )
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
        assumptions.move_to([3.2, 0.55, 0])
        _beat(self, LaggedStart(*(FadeIn(item, shift=UP * 0.1) for item in assumptions), lag_ratio=0.2), duration=2.5)
        phase = ValueTracker(0.0)

        def move_particles(mob):
            for particle, rest in zip(mob, rest_positions):
                angle = 2 * PI * ((rest[0] + 6) / 2.4 - phase.get_value())
                particle.move_to(rest + RIGHT * (0.11 * np.sin(angle)))
                particle.set_color(interpolate_color(BLUE_B, RED_C, (1 - np.cos(angle)) / 2))

        particles.add_updater(move_particles)
        try:
            _beat(self, phase.animate.set_value(3), duration=6)
        finally:
            particles.remove_updater(move_particles)
        conclusion = _text("Keep the pressure-wave physics. Leave out shear, attenuation, and anisotropy.",
                           23, SOURCE_COLOR)
        conclusion.move_to([0, -2.55, 0])
        qualification = _text("Useful for a first model, not a complete description of elastic rock.", 19, GRAY_B)
        qualification.move_to([0, -3.2, 0])
        _beat(self, FadeIn(conclusion), FadeIn(qualification))
        _hold(self, 3)


def _wavelet_graph(center, width=4.8, height=1.8):
    frequency = wave.SOURCE_FREQUENCY
    axes = Axes(
        x_range=[0, 3 / frequency, 1 / frequency], y_range=[-0.6, 1.2, 0.6],
        x_length=width, y_length=height, tips=False,
        axis_config={"stroke_width": 1.2, "include_numbers": False},
    ).move_to(center)
    curve = axes.plot(lambda time: ricker_wavelet(time, frequency),
                      x_range=[0, 3 / frequency], color=SOURCE_COLOR, stroke_width=3)
    return axes, curve


class AcousticWaveEquation(Scene):
    """The scalar pressure PDE and the inputs needed for a forward problem."""

    def construct(self):
        _heading(self, 3, "The acoustic wave equation")
        equation = MathTex(
            r"\frac{\partial^2 p}{\partial t^2}", "=", r"c^2(\mathbf{x})",
            r"\nabla^2 p", "+", r"s(t)\,\delta(\mathbf{x}-\mathbf{x}_s)",
            font_size=43,
        )
        if equation.width > 12:
            equation.scale_to_fit_width(12)
        equation.move_to([0, 1.65, 0])
        equation[2].set_color(MODEL_COLOR)
        equation[5].set_color(SOURCE_COLOR)
        definition = _text("p(x, z, t): pressure disturbance     c(x, z): P-wave speed", 22, GRAY_B)
        definition.move_to([0, 0.55, 0])
        _beat(self, Write(equation), duration=2)
        _beat(self, FadeIn(definition))
        _hold(self, 2)

        source_label = _text("Source wavelet s(t)", 24, SOURCE_COLOR).move_to([-3.4, -0.3, 0])
        axes, curve = _wavelet_graph([-3.4, -1.6, 0], width=4.8, height=1.65)
        velocity = wave.build_velocity_model()
        colors = np.array([[30, 94, 139], [46, 151, 139], [202, 169, 80]], dtype=np.uint8)
        layer_index = np.zeros(velocity.shape, dtype=int)
        layer_index[wave.LAYER_2_START:wave.LAYER_3_START] = 1
        layer_index[wave.LAYER_3_START:] = 2
        model_image = ImageMobject(colors[layer_index]).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        model_image.set_height(2).move_to([3.5, -1.6, 0])
        model_label = _text("Candidate wave speed c(x, z)", 24, MODEL_COLOR).move_to([3.5, -0.3, 0])
        model_values = _text("2000 / 4000 / 5000 m/s", 18, GRAY_B).next_to(model_image, DOWN, buff=0.15)
        _beat(self, Circumscribe(equation[5], color=SOURCE_COLOR), FadeIn(source_label),
              FadeIn(axes), Create(curve), duration=1.8)
        _beat(self, Circumscribe(equation[2], color=MODEL_COLOR), FadeIn(model_label),
              FadeIn(model_image), FadeIn(model_values), duration=1.8)
        conditions = _text("Also specify receiver/source geometry, initial state, and boundary conditions.", 20, GRAY_B)
        conditions.move_to([0, -3.25, 0])
        _beat(self, FadeIn(conditions))
        _hold(self, 3)


class WaveEquationPhysics(
    SpringOscillationScene,
    CoupledOscillationScene,
    LargeCoupledOscillationScene,
    StiffLayerOscillationScene,
    Grid2DOscillationScene,
):
    """Play the repository's five existing oscillator scenes as one chapter."""

    def construct(self):
        SpringOscillationScene.construct(self)
        self.clear()
        CoupledOscillationScene.construct(self)
        self.clear()
        LargeCoupledOscillationScene.construct(self)
        self.clear()
        StiffLayerOscillationScene.construct(self)
        self.clear()
        Grid2DOscillationScene.construct(self)


class SourceEstimation(Scene):
    """Distinguish source measurement/estimation from the chosen Ricker model."""

    def construct(self):
        _heading(self, 5, "Where does the source wavelet come from?")
        premise = _text("Measure it, or estimate it using assumptions about propagation.", 23, GRAY_B)
        premise.move_to([0, 2.2, 0])
        _beat(self, FadeIn(premise))
        axes, curve = _wavelet_graph([-3.4, 0.55, 0], width=4.9, height=1.7)
        frequency = _text(f"Ricker model: {wave.SOURCE_FREQUENCY:g} Hz", 23, SOURCE_COLOR)
        frequency.next_to(axes, UP, buff=0.18)
        _beat(self, FadeIn(axes), FadeIn(frequency), Create(curve), duration=2)

        routes = VGroup(
            _text("Measure: source monitor / near-field sensor", 22, width=5.6),
            _text("Vibroseis: known pilot sweep + correlation", 22, width=5.6),
            _text("Estimate: direct arrivals + propagation model", 22, width=5.6),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([3.3, 0.7, 0])
        _beat(self, LaggedStart(*(FadeIn(route) for route in routes), lag_ratio=0.25), duration=2.5)
        gaussian = MathTex(
            r"s_{\mathrm{Ricker}}(t)\;\propto\;-\frac{d^2}{dt^2}"
            r"\exp\!\left[-\big(\pi f_0(t-t_0)\big)^2\right]",
            font_size=32, color=SOURCE_COLOR,
        ).move_to([0, -1.1, 0])
        _beat(self, Write(gaussian), duration=1.5)
        convolution = MathTex(r"d_i(t)=s(t)*g_i(t;c)+n_i(t)", font_size=34).move_to([0, -2.05, 0])
        relation = _text("recording = source convolved with earth response + noise", 19, GRAY_B)
        relation.move_to([0, -2.65, 0])
        _beat(self, Write(convolution), FadeIn(relation), duration=1.5)
        caveat = _text("This demo prescribes a Ricker wavelet; it is not recovered from these ten traces.",
                       20, SOURCE_COLOR)
        caveat.move_to([0, -3.3, 0])
        _beat(self, FadeIn(caveat))
        _hold(self, 3)


class VelocityEstimation(Scene):
    """Illustrate moveout and the constraints needed to estimate wave speed."""

    def construct(self):
        _heading(self, 6, "How do we estimate wave speed?")
        illustration = _text("Illustrative reflection moveout", 23, MODEL_COLOR).move_to([-3.4, 2.15, 0])
        axes = Axes(
            x_range=[0, 1.3, 0.3], y_range=[0, 0.9, 0.3],
            x_length=4.7, y_length=2.25, tips=False,
            axis_config={"include_numbers": True, "font_size": 16, "stroke_width": 1.2},
        ).move_to([-3.4, 0.5, 0])
        offset_label = _text("Offset (km)", 18).next_to(axes, DOWN, buff=0.15)
        time_label = _text("Travel time (s)", 18).rotate(PI / 2).next_to(axes, LEFT, buff=0.15)
        positions = wave.ReceiverTraces()._trace_receiver_positions()
        offsets = np.array([col - wave.SOURCE_COL for _, col in positions]) * wave.GRID_SPACING / 1000
        zero_offset_time, rms_velocity = 0.35, 2500.0
        times = hyperbolic_traveltime(zero_offset_time, rms_velocity, offsets * 1000)
        picks = VGroup(*(Dot(axes.c2p(offset, time), radius=0.045, color=MODEL_COLOR)
                         for offset, time in zip(offsets, times)))
        trial = axes.plot(lambda offset: hyperbolic_traveltime(zero_offset_time, 1800.0, offset * 1000),
                          x_range=[0, 1.25], color=SOURCE_COLOR, stroke_width=2)
        fitted = axes.plot(lambda offset: hyperbolic_traveltime(zero_offset_time, rms_velocity, offset * 1000),
                           x_range=[0, 1.25], color=SOURCE_COLOR, stroke_width=2)
        _beat(self, FadeIn(illustration), FadeIn(axes), FadeIn(offset_label), FadeIn(time_label))
        _beat(self, FadeIn(picks), Create(trial), duration=1.5)
        _beat(self, Transform(trial, fitted), duration=2)
        moveout = MathTex(r"t^2(x)=t_0^2+\frac{x^2}{v_{\mathrm{RMS}}^2}", font_size=29)
        moveout.move_to([-3.4, -1.75, 0])
        schematic = _text("Synthetic illustration, not picks from these recordings.", 16, GRAY_B, width=5.9)
        schematic.move_to([-3.4, -2.35, 0])
        _beat(self, Write(moveout), FadeIn(schematic))

        routes = VGroup(*[
            _text(content, 23, width=5.5) for content in (
                "First arrivals: travel-time tomography",
                "Moveout: RMS velocity, not interval velocity",
                "Well logs / checkshots: local constraints",
                "Refine: match modeled and recorded waves",
            )
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([3.25, 0.6, 0])
        _beat(self, LaggedStart(*(FadeIn(route) for route in routes), lag_ratio=0.2), duration=3)
        objective = MathTex(
            r"\min_c\;\sum_i\left\|p_c(\mathbf{x}_i,t)-d_i(t)\right\|_2^2"
            r"+\lambda\,\mathcal{R}[c]", font_size=31,
        ).move_to([2.1, -2.95, 0])
        _beat(self, Write(objective), duration=1.8)
        caveat = _text("An estimated source, prior constraints, and more data are needed; ten traces do not fix a unique 2D earth.",
                       18, GRAY_B)
        caveat.move_to([0, -3.6, 0])
        _beat(self, FadeIn(caveat))
        _hold(self, 3)