# animations
Manim animations for various projects

## Notes

* There are two versions of manim: Manim-CE and ManimGL (the latter is the original used by 3b1b)
Only ManimGL supports interactive rendering which is super useful for debugging animations.
Therefore for complex animations ManimGL is recommended.
  
* In manimgl for regular Latex use TexText() and for math mode use just Tex(). This is different from manim-ce.

* To set breakpoints and debug complex scripts you can use run.py target (i.e. in PyCharm).
Example command line (similar to just running manim from command line):
`render icra2023_video_ce.py Title -q h -p`

## Installation

```shell
# for manim-ce:
pip install manim

# for latex:
sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-generic-recommended

# or full installation:
sudo apt install texlive-full
```
