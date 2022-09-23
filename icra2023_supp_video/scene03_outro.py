"""
This is the version for manim community edition (Manim CE).
"""

from manim import *

DEFAULT_WAIT = 2.5


class Outro(Scene):
    def construct(self):
        text = (
            r"Thank you for watching!\\"
            r"Please find additional information at:\\"
            r"https://sites.google.com/view/dexpbt"
        )

        tex = Tex(text, font_size=48)
        # ul = Underline(tex)
        tex.scale_to_fit_width(0.9 * config.frame_width)
        self.play(FadeIn(tex))
        self.wait(DEFAULT_WAIT)
        # self.play(FadeOut(tex))
        self.wait(0.5)
