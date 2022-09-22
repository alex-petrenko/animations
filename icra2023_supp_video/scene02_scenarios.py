"""
This is the version for manim community edition (Manim CE).
"""

from manim import *

DEFAULT_WAIT = 2.5


class Scenarios(Scene):
    def construct(self):
        scenarios = [
            r"\textbf{Scenario $\#1$:} Single-arm regrasping.",
            r"\textbf{Scenario $\#2$:} Single-arm throwing.",
            r"\textbf{Scenario $\#3$:} Single-arm reorientation.",
            r"\textbf{Scenario $\#4$:} Dual-arm regrasping.",
            r"\textbf{Scenario $\#5$:} Dual-arm reorientation.",
        ]

        for i_, scenario in enumerate(scenarios):
            tex = Tex(scenario, font_size=48)
            tex.scale_to_fit_width(0.9 * config.frame_width)
            self.play(FadeIn(tex))
            self.wait(DEFAULT_WAIT)
            self.play(FadeOut(tex))
            self.wait(0.05)
