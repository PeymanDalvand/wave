"""Derivation of the Wiener-Hopf prediction error filter (PEF).

Purely a symbolic/didactic animation -- it walks through the exact linear
system (``R f = g``, a Toeplitz autocorrelation matrix) that
``seismic_lib.design_pef`` solves numerically via ``scipy.linalg.solve_toeplitz``
for the ``PredictiveDeconvolution`` scene.
"""

from manim import *


class PEFConstruction(Scene):
    """Animates the three steps of building a prediction error filter."""

    def construct(self):
        title = Text("Constructing the Prediction Error Filter (PEF)", font_size=32).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        r_array = self._step_1_autocorrelation_lags()
        equation, labels = self._step_2_wiener_hopf_equations(r_array)
        self._step_3_assemble_pef_array(equation)

    def _step_1_autocorrelation_lags(self) -> MathTex:
        """Extract the autocorrelation lags used to build the normal equations."""
        step1_text = Text("1. Extract Autocorrelation Lags", font_size=20, color=BLUE).to_edge(LEFT, buff=1.0).shift(UP * 2)

        # r_0 is the zero-lag (maximum energy), r_alpha is the lag where the multiple arrives
        r_array = MathTex("r", "=", "[", "r_0", ",", "r_1", ",", "r_2", ",", "\\dots", ",", "r_\\alpha", ",", "\\dots", "]")
        r_array.next_to(step1_text, DOWN, aligned_edge=LEFT, buff=0.3)

        self.play(Write(step1_text), FadeIn(r_array))
        self.wait(1)
        return r_array

    def _step_2_wiener_hopf_equations(self, r_array: MathTex) -> tuple[VGroup, MathTex]:
        """Formulate the Wiener-Hopf normal equations R f = g as a Toeplitz system."""
        step2_text = Text("2. Formulate Wiener-Hopf Equations", font_size=20, color=BLUE).next_to(r_array, DOWN, buff=0.7, aligned_edge=LEFT)

        # The Autocorrelation Matrix R (Toeplitz structure - diagonals are identical)
        matrix_R = Matrix([
            ["r_0", "r_1", "r_2"],
            ["r_1", "r_0", "r_1"],
            ["r_2", "r_1", "r_0"]
        ])

        # The unknown filter coefficients f
        vector_f = Matrix([["f_0"], ["f_1"], ["f_2"]])

        # The target cross-correlation vector g (shifted by the prediction distance)
        vector_g = Matrix([["r_\\alpha"], ["r_{\\alpha+1}"], ["r_{\\alpha+2}"]])

        equals = MathTex("=")

        equation = VGroup(matrix_R, vector_f, equals, vector_g).arrange(RIGHT)
        equation.next_to(step2_text, DOWN, aligned_edge=LEFT, buff=0.3)

        labels = MathTex("\\mathbf{R}", "\\mathbf{f}", "=", "\\mathbf{g}", color=YELLOW)
        labels[0].next_to(matrix_R, DOWN)
        labels[1].next_to(vector_f, DOWN)
        labels[3].next_to(vector_g, DOWN)

        self.play(Write(step2_text))
        self.play(Write(equation), FadeIn(labels))
        self.wait(2)
        return equation, labels

    def _step_3_assemble_pef_array(self, equation: VGroup) -> None:
        """Assemble the final filter operator applied to the seismic trace."""
        step3_text = Text("3. Assemble the Final Filter Array", font_size=20, color=BLUE).next_to(equation, DOWN, buff=0.7, aligned_edge=LEFT)

        # The final mathematical operator applied to the seismic trace
        pef_array = MathTex(
            "\\text{PEF}", "=", "[",
            "1", ",",
            "0", ",", "\\dots", ",", "0", ",",
            "-f_0", ",", "-f_1", ",", "-f_2", "]"
        )
        pef_array.next_to(step3_text, DOWN, aligned_edge=LEFT, buff=0.5).shift(RIGHT * 1.5)

        # Highlight the primary pass-through
        brace_primary = Brace(pef_array[3:4], UP, color=GREEN)
        text_primary = brace_primary.get_text("Pass Primary").set_color(GREEN).scale(0.6)

        # Highlight the physical prediction gap
        brace_lag = Brace(pef_array[5:11], DOWN, color=GRAY)
        text_lag = brace_lag.get_text(r"Prediction Gap ($\alpha$)").set_color(GRAY).scale(0.6)

        # Highlight the active subtraction filter
        brace_filter = Brace(pef_array[11:16], DOWN, color=RED)
        text_filter = brace_filter.get_text("Subtraction Coefficients").set_color(RED).scale(0.6)

        self.play(Write(step3_text))
        self.play(Write(pef_array))
        self.play(
            FadeIn(brace_primary), FadeIn(text_primary),
            FadeIn(brace_lag), FadeIn(text_lag),
            FadeIn(brace_filter), FadeIn(text_filter)
        )
        self.wait(4)