from manim import *
import numpy as np


def make_spring_vertices(p1: np.ndarray, p2: np.ndarray, coils: int = 8, radius: float = 0.12) -> list[np.ndarray]:
    """Generate 3D vertices for a dynamic zig-zag spring between p1 and p2."""
    diff = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-4:
        return [p1, p2]
    t_hat = diff / length
    n_hat = np.array([-t_hat[1], t_hat[0], 0.0])

    pts = [p1, p1 + 0.1 * diff]
    n_peaks = 2 * coils
    for k in range(n_peaks):
        frac = 0.1 + 0.8 * (k + 0.5) / n_peaks
        sign = 1.0 if k % 2 == 0 else -1.0
        pts.append(p1 + frac * diff + sign * radius * n_hat)
    pts.extend([p1 + 0.9 * diff, p2])
    return pts


class CoupledChainSolver:
    """
    1D coupled mass-spring chain solver derived from macroscopic (rho, kappa)
    so that wave speed c = sqrt(kappa / rho) remains invariant as dx -> 0.
    """

    def __init__(self, n_masses: int, length: float = 10.0, rho: float = 1.0, kappa: float = 6.25, area: float = 1.0):
        self.N = n_masses
        self.L = length
        self.dx = length / n_masses
        self.rho = rho
        self.kappa = kappa
        self.area = area
        self.c = np.sqrt(kappa / rho)

        # Microscopic lumped parameters mapped from macroscopic continuum properties
        self.m = rho * area * self.dx
        self.K = kappa * area / self.dx

        self.x_eq = np.linspace(0.0, length, n_masses)
        self.psi = np.zeros(n_masses, dtype=np.float64)
        self.vel = np.zeros(n_masses, dtype=np.float64)
        self.acc = np.zeros(n_masses, dtype=np.float64)
        self.t = 0.0

        # Absorbing sponge damping at the right boundary to prevent reflections
        self.damping = np.zeros(n_masses, dtype=np.float64)
        sponge_len = max(3, int(0.22 * n_masses))
        for idx in range(n_masses - sponge_len, n_masses):
            s = (idx - (n_masses - sponge_len)) / sponge_len
            self.damping[idx] = 8.0 * (s ** 2)

    def step(self, dt: float, source_fn=None, substeps: int = 12):
        dt_sub = dt / substeps
        k_over_m = self.K / self.m

        for _ in range(substeps):
            self.t += dt_sub
            if source_fn is not None:
                self.psi[0] = source_fn(self.t)
                self.vel[0] = 0.0

            # F = ma on interior masses: K * (psi_{i+1} - 2*psi_i + psi_{i-1})
            self.acc[1:-1] = (
                k_over_m * (self.psi[2:] - 2.0 * self.psi[1:-1] + self.psi[:-2])
                - self.damping[1:-1] * self.vel[1:-1]
            )
            self.vel[1:-1] += self.acc[1:-1] * dt_sub
            self.psi[1:-1] += self.vel[1:-1] * dt_sub

    def get_pressure(self) -> np.ndarray:
        """Compute acoustic pressure p = -kappa * d(psi)/dx via central differences."""
        dpsi_dx = np.gradient(self.psi, self.dx)
        return -self.kappa * dpsi_dx


class MassSpringToAcousticWave(Scene):
    def construct(self):
        self.camera.background_color = "#0e1117"

        self.act_1_single_oscillator()
        self.clear_stage()

        self.act_2_discrete_coupled_chain()
        self.clear_stage()

        self.act_3_continuum_limit()
        self.clear_stage()

        self.act_4_displacement_to_pressure()

    def clear_stage(self):
        for m in self.mobjects:
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    # -------------------------------------------------------------------------
    # ACT 1: Single Mass-Spring Oscillator (Temporal Only)
    # -------------------------------------------------------------------------
    def act_1_single_oscillator(self):
        title = Text("1. Single Oscillator: Temporal Vibration Only", font_size=28, color=BLUE_B).to_edge(UP, buff=0.35)
        eq = MathTex(
            r"F = ma \implies m\frac{d^2\psi}{dt^2} = -K\psi",
            font_size=34
        ).next_to(title, DOWN, buff=0.2)

        # Physical anchor wall & equilibrium line
        wall_x = -5.2
        eq_x = -2.2
        center_y = -0.3

        wall = Line([wall_x, center_y - 1.0, 0], [wall_x, center_y + 1.0, 0], stroke_width=6, color=GREY_B)
        eq_dashed = DashedLine([eq_x, center_y - 1.4, 0], [eq_x, center_y + 1.4, 0], color=GREY_C, stroke_opacity=0.6)
        eq_label = MathTex(r"x_0", font_size=26, color=GREY_C).next_to(eq_dashed, DOWN, buff=0.15)

        state = {"psi": 1.1, "v": 0.0, "t": 0.0, "omega": 3.2, "history": []}

        mass_box = Square(side_length=0.75, fill_color=BLUE_D, fill_opacity=0.9, stroke_color=WHITE, stroke_width=2)
        mass_box.move_to([eq_x + state["psi"], center_y, 0])
        mass_label = MathTex("m", font_size=28).move_to(mass_box.get_center())

        spring = VMobject(color=GREY_A, stroke_width=3)
        spring.set_points_as_corners(
            make_spring_vertices(np.array([wall_x, center_y, 0]), mass_box.get_left(), coils=7, radius=0.18)
        )

        force_arrow = Line(ORIGIN, RIGHT * 0.1, color=RED_C, stroke_width=6).add_tip(tip_length=0.2)
        force_label = MathTex(r"F = -K\psi", font_size=26, color=RED_C)

        # Time-history axes on the right
        ax = Axes(
            x_range=[0, 6, 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=5.5,
            y_length=3.0,
            axis_config={"include_tip": False, "color": GREY_B},
        ).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.3)
        ax_labels = VGroup(
            MathTex("t", font_size=26).next_to(ax.x_axis, RIGHT, buff=0.15),
            MathTex(r"\psi(t)", font_size=26).next_to(ax.y_axis, UP, buff=0.15),
        )
        trace_curve = VMobject(color=YELLOW_C, stroke_width=3.5)

        def update_single(mob, dt):
            sub = 8
            dt_s = dt / sub
            for _ in range(sub):
                acc = -(state["omega"] ** 2) * state["psi"]
                state["v"] += acc * dt_s
                state["psi"] += state["v"] * dt_s
                state["t"] += dt_s

            curr_x = eq_x + state["psi"]
            mass_box.move_to([curr_x, center_y, 0])
            mass_label.move_to(mass_box.get_center())
            spring.set_points_as_corners(
                make_spring_vertices(np.array([wall_x, center_y, 0]), mass_box.get_left(), coils=7, radius=0.18)
            )

            # Force vector proportional to -psi
            f_val = -0.85 * state["psi"]
            start_pt = np.array([curr_x, center_y + 0.65, 0])
            if abs(f_val) > 0.12:
                end_pt = start_pt + np.array([f_val, 0, 0])
            else:
                end_pt = start_pt + np.array([0.12 * np.sign(f_val + 1e-6), 0, 0])
            force_arrow.put_start_and_end_on(start_pt, end_pt)
            force_label.next_to(force_arrow, UP, buff=0.1)

            if state["t"] <= 6.0:
                state["history"].append(ax.c2p(state["t"], state["psi"]))
                if len(state["history"]) > 1:
                    trace_curve.set_points_as_corners(state["history"])

        self.play(Write(title), FadeIn(eq), run_time=0.8)
        self.play(
            Create(wall), Create(eq_dashed), FadeIn(eq_label),
            Create(spring), FadeIn(mass_box), FadeIn(mass_label),
            Create(ax), FadeIn(ax_labels), FadeIn(force_arrow), FadeIn(force_label),
            run_time=0.8
        )
        self.add(trace_curve)
        mass_box.add_updater(update_single)
        self.wait(5.5)
        mass_box.remove_updater(update_single)

    # -------------------------------------------------------------------------
    # ACT 2: Discrete 1D Coupled Chain (Spatial Coupling via Hooke's Law)
    # -------------------------------------------------------------------------
    def act_2_discrete_coupled_chain(self):
        title = Text("2. Coupled 1D Mass-Spring Chain: Spatial Stencil", font_size=28, color=BLUE_B).to_edge(UP, buff=0.35)
        eq = MathTex(
            r"m\frac{d^2\psi_i}{dt^2} = \underbrace{K(\psi_{i+1}-\psi_i)}_{F_{\text{right}}} - \underbrace{K(\psi_i-\psi_{i-1})}_{F_{\text{left}}} = K(\psi_{i+1} - 2\psi_i + \psi_{i-1})",
            font_size=30
        ).next_to(title, DOWN, buff=0.2)

        n_chain = 15
        solver = CoupledChainSolver(n_masses=n_chain, length=11.0, rho=1.0, kappa=5.5)

        chain_y = 0.65
        x_left = -5.5
        scale_disp = 0.55  # Visual scaling for longitudinal displacement
        focus_idx = 6      # Highlighted mass "i"

        springs = VGroup(*[VMobject(color=GREY_B, stroke_width=2.5) for _ in range(n_chain - 1)])
        masses = VGroup()
        for i in range(n_chain):
            col = YELLOW_C if i == focus_idx else BLUE_D
            rad = 0.20 if i == focus_idx else 0.16
            masses.add(Circle(radius=rad, fill_color=col, fill_opacity=0.95, stroke_color=WHITE, stroke_width=1.5))

        i_label = MathTex("m_i", font_size=26, color=YELLOW_C)

        # Lower plot for discrete displacement psi_i
        ax = Axes(
            x_range=[0, 11.0, 1.0],
            y_range=[-1.1, 1.1, 0.5],
            x_length=11.0,
            y_length=2.4,
            axis_config={"include_tip": False, "color": GREY_B},
        ).shift(DOWN * 1.9)
        ax_label = MathTex(r"\psi_i(t)", font_size=26, color=YELLOW_C).next_to(ax.y_axis, UP, buff=0.1)

        stems = VGroup(*[Line(ORIGIN, UP * 0.01, color=BLUE_C, stroke_width=3) for _ in range(n_chain)])
        dots = VGroup(*[Dot(radius=0.06, color=YELLOW_C if i == focus_idx else TEAL_B) for i in range(n_chain)])
        envelope = VMobject(color=TEAL_C, stroke_width=2, stroke_opacity=0.6)

        def source_wavelet(t):
            # Single smooth Gaussian derivative pulse
            t0 = 0.9
            tau = (t - t0) / 0.28
            return -0.85 * tau * np.exp(-0.5 * tau ** 2) if t < 2.5 else 0.0

        def update_chain(mob, dt):
            solver.step(dt, source_fn=source_wavelet, substeps=10)

            for i in range(n_chain):
                px = x_left + solver.x_eq[i] + scale_disp * solver.psi[i]
                masses[i].move_to([px, chain_y, 0])

            i_label.next_to(masses[focus_idx], UP, buff=0.15)

            for i in range(n_chain - 1):
                p1 = masses[i].get_right()
                p2 = masses[i + 1].get_left()
                springs[i].set_points_as_corners(make_spring_vertices(p1, p2, coils=5, radius=0.11))

            pts = []
            for i in range(n_chain):
                base_pt = ax.c2p(solver.x_eq[i], 0)
                top_pt = ax.c2p(solver.x_eq[i], solver.psi[i])
                if np.linalg.norm(top_pt - base_pt) < 1e-3:
                    top_pt = base_pt + np.array([0, 1e-3, 0])
                stems[i].put_start_and_end_on(base_pt, top_pt)
                dots[i].move_to(top_pt)
                pts.append(top_pt)
            envelope.set_points_as_corners(pts)

        update_chain(masses, 0.0)
        self.play(Write(title), FadeIn(eq), run_time=0.8)
        self.play(
            Create(springs), FadeIn(masses), FadeIn(i_label),
            Create(ax), FadeIn(ax_label), Create(stems), FadeIn(dots), Create(envelope),
            run_time=0.8
        )
        masses.add_updater(update_chain)
        self.wait(5.8)
        masses.remove_updater(update_chain)

    # -------------------------------------------------------------------------
    # ACT 3: Continuum Limit (dx -> 0 with invariant c = sqrt(kappa / rho))
    # -------------------------------------------------------------------------
    def act_3_continuum_limit(self):
        title = Text("3. Continuum Limit: Shrinking Grid Spacing (Δx → 0)", font_size=28, color=BLUE_B).to_edge(UP, buff=0.35)
        eq = MathTex(
            r"\rho = \frac{m}{A\Delta x}, \quad \kappa = \frac{K\Delta x}{A} \implies "
            r"\rho\frac{\partial^2\psi}{\partial t^2} = \kappa\frac{\partial^2\psi}{\partial x^2} "
            r"\quad \left(c = \sqrt{\frac{\kappa}{\rho}}\right)",
            font_size=29
        ).next_to(title, DOWN, buff=0.2)

        # Two solvers sharing the EXACT same macroscopic (rho, kappa), differing only in N
        solver_coarse = CoupledChainSolver(n_masses=14, length=10.0, rho=1.0, kappa=6.25)
        solver_fine = CoupledChainSolver(n_masses=100, length=10.0, rho=1.0, kappa=6.25)

        ax_coarse = Axes(
            x_range=[0, 10, 1], y_range=[-1.1, 1.1, 0.5], x_length=10.5, y_length=1.8,
            axis_config={"include_tip": False, "color": GREY_B}
        ).shift(UP * 0.45)
        lbl_coarse = Text("Coarse Discrete Chain (N = 14, Large Δx)", font_size=20, color=ORANGE).next_to(ax_coarse, UP, buff=0.1).align_to(ax_coarse, LEFT)

        ax_fine = Axes(
            x_range=[0, 10, 1], y_range=[-1.1, 1.1, 0.5], x_length=10.5, y_length=1.8,
            axis_config={"include_tip": False, "color": GREY_B}
        ).shift(DOWN * 2.15)
        lbl_fine = Text("Continuum Limit (N = 100, Δx → 0): Smooth PDE Solution", font_size=20, color=TEAL_B).next_to(ax_fine, UP, buff=0.1).align_to(ax_fine, LEFT)

        coarse_line = VMobject(color=ORANGE, stroke_width=2.5)
        coarse_dots = VGroup(*[Dot(radius=0.055, color=ORANGE) for _ in range(solver_coarse.N)])
        fine_curve = VMobject(color=TEAL_B, stroke_width=3.5)

        def ricker_source(t):
            f0 = 1.1
            t0 = 1.0
            arg = (np.pi * f0 * (t - t0)) ** 2
            return 0.9 * (1.0 - 2.0 * arg) * np.exp(-arg) if t < 2.2 else 0.0

        def update_limit(mob, dt):
            solver_coarse.step(dt, source_fn=ricker_source, substeps=12)
            solver_fine.step(dt, source_fn=ricker_source, substeps=16)

            pts_c = [ax_coarse.c2p(solver_coarse.x_eq[i], solver_coarse.psi[i]) for i in range(solver_coarse.N)]
            coarse_line.set_points_as_corners(pts_c)
            for i, pt in enumerate(pts_c):
                coarse_dots[i].move_to(pt)

            pts_f = [ax_fine.c2p(solver_fine.x_eq[i], solver_fine.psi[i]) for i in range(solver_fine.N)]
            fine_curve.set_points_as_corners(pts_f)

        update_limit(fine_curve, 0.0)
        self.play(Write(title), FadeIn(eq), run_time=0.8)
        self.play(
            Create(ax_coarse), FadeIn(lbl_coarse), Create(coarse_line), FadeIn(coarse_dots),
            Create(ax_fine), FadeIn(lbl_fine), Create(fine_curve),
            run_time=0.8
        )
        fine_curve.add_updater(update_limit)
        self.wait(5.5)
        fine_curve.remove_updater(update_limit)

    # -------------------------------------------------------------------------
    # ACT 4: Particle Displacement psi(x,t) -> Acoustic Pressure p(x,t)
    # -------------------------------------------------------------------------
    def act_4_displacement_to_pressure(self):
        title = Text("4. Particle Displacement ψ(x,t) to Acoustic Pressure p(x,t)", font_size=28, color=BLUE_B).to_edge(UP, buff=0.3)
        eq = MathTex(
            r"p(x,t) = -\kappa\frac{\partial\psi}{\partial x} \implies "
            r"\frac{\partial^2 p}{\partial t^2} = c^2\frac{\partial^2 p}{\partial x^2}",
            font_size=30
        ).next_to(title, DOWN, buff=0.15)

        n_slabs = 95
        solver = CoupledChainSolver(n_masses=n_slabs, length=10.0, rho=1.0, kappa=6.25)

        # 1D Acoustic Waveguide duct showing slab bunching (compression) and stretching (rarefaction)
        duct_y = 1.05
        duct_w = 10.5
        duct_h = 0.75
        duct_left = -duct_w / 2.0

        duct_box = SurroundingRectangle(
            Rectangle(width=duct_w, height=duct_h).move_to([0, duct_y, 0]),
            color=GREY_B, buff=0.04, stroke_width=2
        )
        duct_lbl = Text("1D Acoustic Fluid Slabs (Red = Compression p > 0, Blue = Rarefaction p < 0)", font_size=18, color=GREY_A).next_to(duct_box, UP, buff=0.08)

        slabs = VGroup(*[
            Line([0, duct_y - duct_h / 2, 0], [0, duct_y + duct_h / 2, 0], stroke_width=4.5)
            for _ in range(n_slabs)
        ])

        # Synchronized Axes for Displacement psi(x,t) and Pressure p(x,t)
        ax_psi = Axes(
            x_range=[0, 10, 1], y_range=[-1.1, 1.1, 0.5], x_length=duct_w, y_length=1.45,
            axis_config={"include_tip": False, "color": GREY_B}
        ).move_to([0, -0.55, 0])
        lbl_psi = MathTex(r"\text{Displacement } \psi(x,t)", font_size=23, color=YELLOW_C).next_to(ax_psi, UP, buff=0.05).align_to(ax_psi, LEFT)

        ax_p = Axes(
            x_range=[0, 10, 1], y_range=[-1.1, 1.1, 0.5], x_length=duct_w, y_length=1.45,
            axis_config={"include_tip": False, "color": GREY_B}
        ).move_to([0, -2.45, 0])
        lbl_p = MathTex(r"\text{Acoustic Pressure } p(x,t) = -\kappa\frac{\partial\psi}{\partial x}", font_size=23, color=RED_C).next_to(ax_p, UP, buff=0.05).align_to(ax_p, LEFT)

        curve_psi = VMobject(color=YELLOW_C, stroke_width=3.2)
        curve_p = VMobject(color=RED_C, stroke_width=3.2)

        def gaussian_pulse(t):
            t0 = 0.85
            tau = (t - t0) / 0.25
            return 0.85 * np.exp(-0.5 * tau ** 2) if t < 2.2 else 0.0

        def update_acoustic(mob, dt):
            solver.step(dt, source_fn=gaussian_pulse, substeps=16)
            p_field = solver.get_pressure()
            p_norm = np.clip(p_field / 14.0, -1.0, 1.0)

            scale_x = duct_w / solver.L
            disp_visual_gain = 0.32

            pts_psi = []
            pts_p = []

            for i in range(n_slabs):
                # Physical slab position shifted by longitudinal displacement psi
                x_phys = duct_left + (solver.x_eq[i] + disp_visual_gain * solver.psi[i]) * scale_x
                slabs[i].put_start_and_end_on([x_phys, duct_y - duct_h / 2, 0], [x_phys, duct_y + duct_h / 2, 0])

                # Color slab by acoustic pressure perturbation p = -kappa * d(psi)/dx
                val = p_norm[i]
                if val >= 0:
                    col = interpolate_color(GREY_D, RED_B, float(val))
                else:
                    col = interpolate_color(GREY_D, BLUE_B, float(-val))
                slabs[i].set_color(col)

                pts_psi.append(ax_psi.c2p(solver.x_eq[i], solver.psi[i]))
                pts_p.append(ax_p.c2p(solver.x_eq[i], p_norm[i]))

            curve_psi.set_points_as_corners(pts_psi)
            curve_p.set_points_as_corners(pts_p)

        update_acoustic(slabs, 0.0)
        self.play(Write(title), FadeIn(eq), run_time=0.8)
        self.play(
            Create(duct_box), FadeIn(duct_lbl), Create(slabs),
            Create(ax_psi), FadeIn(lbl_psi), Create(curve_psi),
            Create(ax_p), FadeIn(lbl_p), Create(curve_p),
            run_time=0.8
        )
        slabs.add_updater(update_acoustic)
        self.wait(5.8)
        slabs.remove_updater(update_acoustic)