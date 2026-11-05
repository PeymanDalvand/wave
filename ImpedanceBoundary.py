"""Reflection and transmission of a pulse at an acoustic-impedance boundary.

Two identical Ricker-driven transverse chains are compared: one with a
uniform mass (no impedance contrast, 100% transmission) and one with a
step change in mass (a stand-in for a density/impedance contrast), which
partially reflects and partially transmits the incoming pulse. Physics
lives in ``seismic_lib.DrivenTransverseChain1D``.
"""

from manim import *
import numpy as np

from seismic_lib import DrivenTransverseChain1D, reflection_transmission_coefficients, ricker_wavelet

NUM_PARTICLES = 60
PARTICLE_SPACING = 0.2
STIFFNESS = 300.0
DAMPING = 0.05
HEAVY_MASS_RATIO = 5.0  # mass multiplier past the impedance boundary
SOURCE_FREQUENCY = 1.5
SOURCE_AMPLITUDE = 1.5
SIM_TIME_STEP = 0.002


class ImpedanceBoundary(Scene):
    """Uniform layer (top) vs. a hard-rock impedance boundary (bottom)."""

    def construct(self):
        # 1. Physics: two chains, identical except for a mass step at the midpoint
        mass_uniform = np.ones(NUM_PARTICLES)
        mass_stepped = np.ones(NUM_PARTICLES)
        mass_stepped[NUM_PARTICLES // 2 :] = HEAVY_MASS_RATIO

        chain_uniform = DrivenTransverseChain1D(mass=mass_uniform, stiffness=STIFFNESS, damping=DAMPING)
        chain_stepped = DrivenTransverseChain1D(mass=mass_stepped, stiffness=STIFFNESS, damping=DAMPING)

        def source(t):
            return ricker_wavelet(t, frequency=SOURCE_FREQUENCY, amplitude=SOURCE_AMPLITUDE)

        # Schematic impedance ratio (Z ~ sqrt(mass) for fixed stiffness/spacing)
        reflection, transmission = reflection_transmission_coefficients(1.0, np.sqrt(HEAVY_MASS_RATIO))

        # 2. Visual Elements Setup
        start_x = -(NUM_PARTICLES * PARTICLE_SPACING) / 2

        title1 = Text("Uniform Rock Layer (100% Transmission)", font_size=24, color=BLUE_B).to_edge(UP, buff=0.5)
        dots1 = VGroup(*[Dot(radius=0.06, color=BLUE) for _ in range(NUM_PARTICLES)])
        line1 = VMobject(stroke_color=GRAY_B, stroke_width=2)

        title2 = Text("Hard Rock Boundary (Reflection & Transmission)", font_size=24, color=RED_B).move_to(DOWN * 0.2)
        dots2 = VGroup(*[
            Dot(radius=0.06 if mass_stepped[i] == 1.0 else 0.12, color=BLUE if mass_stepped[i] == 1.0 else RED)
            for i in range(NUM_PARTICLES)
        ])
        line2 = VMobject(stroke_color=GRAY_B, stroke_width=2)

        # Dashed line showing exactly where the geological boundary occurs
        boundary_x = start_x + (NUM_PARTICLES // 2) * PARTICLE_SPACING
        boundary_line = DashedLine(DOWN * 3.5, UP * 3.5, color=WHITE, stroke_opacity=0.5).move_to([boundary_x, 0, 0])
        boundary_label = Text("Impedance Boundary", font_size=16).next_to(boundary_line, UP, buff=0.1)
        coefficients_label = Text(
            f"R \u2248 {reflection:.2f}   T \u2248 {transmission:.2f}", font_size=16, color=YELLOW
        ).next_to(boundary_label, UP, buff=0.1)

        self.add(title1, dots1, line1, title2, dots2, line2, boundary_line, boundary_label, coefficients_label)

        # 3. The Physics-Driven Update Loop
        def update_wave(mob, dt):
            chain_uniform.step(dt, source, dt_sim=SIM_TIME_STEP)
            chain_stepped.step(dt, source, dt_sim=SIM_TIME_STEP)

            for i in range(NUM_PARTICLES):
                dots1[i].move_to([start_x + i * PARTICLE_SPACING, 2.0 + chain_uniform.position[i], 0])
                dots2[i].move_to([start_x + i * PARTICLE_SPACING, -2.5 + chain_stepped.position[i], 0])

            line1.set_points_as_corners([d.get_center() for d in dots1])
            line2.set_points_as_corners([d.get_center() for d in dots2])

        # Attach updater to a dummy Mobject to drive the physics loop
        updater_mob = VMobject()
        updater_mob.add_updater(update_wave)
        self.add(updater_mob)

        # Run long enough to see the wave hit the boundary, split, and reflect back to the start
        self.wait(9)