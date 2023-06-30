"""
This is the version for manim community edition (Manim CE).
"""

from manim import *

DEFAULT_WAIT = 5


class Title(Scene):
    def construct(self):
        title_str = (
            r"DexPBT: Scaling up Dexterous \\ Manipulation for Hand-Arm Systems \\ with Population Based Training"
        )
        title = Tex(title_str, font_size=72)
        title.shift(0.5 * UP)
        self.add(title)
        self.wait(1)

        author_str = r"""
          Aleksei Petrenko$^{1,2}$ Arthur Allshire$^{1,3}$ Gavriel State$^1$\\
          Ankur Handa$^1$ Viktor Makoviychuk$^1$\\
          \vspace{0.5em}
          $^1$NVIDIA, $^2$University of Southern California, $^3$University of Toronto\\
        """
        author = Tex(author_str, font_size=32)
        author.shift(DOWN * 1.5)
        author.scale_to_fit_width(0.9 * config.frame_width)

        icra_str = r"ICRA 2023 Submission"
        icra = Tex(icra_str, font_size=32)
        # align to the bottom
        icra.to_edge(DOWN, buff=0.5)

        self.play(title.animate.shift(UP), FadeIn(author), FadeIn(icra))

        self.wait(DEFAULT_WAIT)

        # fade out everything
        self.play(FadeOut(title), FadeOut(author), FadeOut(icra))
        self.wait(0.05)

        intro_str = [
            r"\textbf{Task:} dexterous object manipulation with a 23-DOF robotic hand-arm system (Allegro hand + KUKA arm).",
            r"\textbf{Approach:} vectorized GPU-accelerated simulation, end-to-end Deep Reinforcement Learning, Population-Based Training.",
        ]
        intro1, intro2 = intro = VGroup(
            Tex(intro_str[0], font_size=48),
            Tex(intro_str[1], font_size=48),
        )

        intro.arrange(DOWN, buff=MED_LARGE_BUFF)
        intro.scale_to_fit_width(0.9 * config.frame_width)

        self.play(FadeIn(intro1))
        self.wait(DEFAULT_WAIT * 2)
        self.play(FadeIn(intro2))
        self.wait(DEFAULT_WAIT * 2)

        self.play(FadeOut(intro))
        self.wait(0.05)
