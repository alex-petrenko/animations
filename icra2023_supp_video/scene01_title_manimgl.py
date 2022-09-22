from manimlib import *
import numpy as np


class Title(Scene):
    CONFIG = {
        "camera_config": {"background_color": "#000000"},
    }

    def construct(self):
        title_str = r"DexPBT: Scaling up Dexterous \\ Manipulation for Hand-Arm Systems \\ with Population Based Training"
        title = TexText(title_str, font_size=72)
        self.add(title)
        self.wait(1)

        author_str = r"""
          Aleksei Petrenko$^{1,2}$ Arthur Allshire$^{1,3}$ Gavriel State$^1$ Ankur Handa$^1$ Viktor Makoviychuk$^1$\\
          \vspace{0.5em}
          $^1$NVIDIA, $^2$University of Southern California, $^3$University of Toronto"""
        author = TexText(author_str, font_size=28)
        author.shift(DOWN*2)

        self.play(title.animate.shift(UP), FadeIn(author))

        self.wait(5)

        # fade out everything
        self.play(FadeOut(title), FadeOut(author))
        self.wait(0.1)


