import random

from manim import *


class PBT(Scene):
    # noinspection PyTypeChecker
    def construct(self):
        """
        Animation explaining the mechanism of population based training.
        """

        random.seed(43)

        final_anim = True

        # add 8 boxes in a 2x4 grid. We only want a border around the outside, and transparent backgrounds
        # these designate the shared folders for 8 policies in the PBT population
        # now adding the boxes by direcly using the manim API:
        # https://docs.manim.community/en/stable/reference/manim.mobject.geometry.Rectangle.html

        boxes = []
        fnames = []

        for i in range(2):
            for j in range(4):
                box = Rectangle(color=WHITE, height=1.6, width=3, fill_opacity=0)  # No fill, only border

                # Move to the proper position, centered on screen
                box.shift(i * 2.5 * DOWN + j * 3.65 * RIGHT - 5.5 * RIGHT + 2 * UP)
                text = Tex(f"/pbt/policy{i * 4 + j}", color=WHITE, font_size=30)
                text.next_to(box, UP)  # Position text above box

                boxes.append(box)
                fnames.append(text)

        def wait(t):
            if final_anim:
                self.wait(t)

        if final_anim:
            self.play(*[Create(box) for box in boxes], *[Write(text) for text in fnames])
        else:
            self.add(*boxes, *fnames)
        wait(0.1)

        # simulate how new checkpoints are added to each folder in the population

        delays = [0, 1, 1, 0, 1, 1, 2, 0]
        chosen_policy = 5
        delays[chosen_policy] = 1

        fitness = [random.random() for _ in range(8)]

        checkpoints = [[] for _ in range(8)]
        for iteration in range(4):
            for p in range(8):
                if delays[p] > iteration:
                    continue

                improvement = random.uniform(-0.1, 3)
                fitness[p] += improvement
                fitness[p] = max(0.1, fitness[p])

            # make sure the chosen policy is one of the worst
            while True:
                argmin = np.argmin(fitness)
                if argmin != chosen_policy:
                    fitness[chosen_policy] -= random.random()
                    fitness[chosen_policy] = max(0, fitness[chosen_policy])
                else:
                    break

            iter_checkpoints = []
            for p in range(8):
                if delays[p] > iteration:
                    continue

                # generate checkpoint texts inside folder boxes
                this_iter = iteration - delays[p]
                text = Tex(f"p{p}-iter{this_iter:02d}-obj{fitness[p]:.2f}.pth", color=WHITE, font_size=20)
                text.next_to(boxes[p], DOWN, buff=-0.6 + this_iter * 0.35, aligned_edge=UP)
                iter_checkpoints.append(text)
                checkpoints[p].append(text)

            # add the new checkpoints to the scene
            if final_anim:
                self.play(*[Write(text) for text in iter_checkpoints])
            else:
                self.add(*iter_checkpoints)
            wait(0.5)

        # highlight animation for the chosen policy
        if final_anim:
            self.play(boxes[chosen_policy].animate.set_color(YELLOW), run_time=0.5)
            self.play(Indicate(checkpoints[chosen_policy][-1], scale_factor=1.1))
        wait(1)

        # select corresponding checkpoint from other policies
        selected_checkpoints = []
        for p in range(8):
            idx = len(checkpoints[chosen_policy]) - 1
            if idx >= len(checkpoints[p]):
                idx = len(checkpoints[p]) - 1

            selected_checkpoints.append(checkpoints[p][idx])

        # highlight animation for the selected checkpoints
        if final_anim:
            self.play(*[text.animate.set_color(YELLOW) for text in selected_checkpoints], run_time=0.5)
            self.play(*[Indicate(text, scale_factor=1.1) for text in selected_checkpoints], run_time=0.5)
        else:
            for c in selected_checkpoints:
                c.set_color(YELLOW)

        wait(1)

        checkpoint_fadeouts = []
        for p in range(8):
            for checkpoint in checkpoints[p]:
                if checkpoint not in selected_checkpoints:
                    checkpoint_fadeouts.append(checkpoint)

        if final_anim:
            self.play(*[FadeOut(t) for t in fnames])
            self.play(*[FadeOut(b) for b in boxes])
            self.play(*[FadeOut(c) for c in checkpoint_fadeouts])
        else:
            self.remove(*fnames)
            self.remove(*boxes)
            self.remove(*checkpoint_fadeouts)

        headers = ["Policy index", "Checkpoint", "Objective"]
        headers = [Tex(h, color=WHITE, font_size=30) for h in headers]
        header_texts = VGroup(*headers)
        header_texts.arrange(RIGHT, buff=1.6)
        header_texts.to_corner(UL, buff=1)

        # add the table
        rows = []
        checkpoint_placeholders = []
        objective_values = []
        for p in range(8):
            policy_idx = Tex(f"Policy \\#{p}", color=WHITE, font_size=24)
            objective_value: float = fitness[p]
            objective_values.append(objective_value)
            objective_value = Tex(f"{objective_value:.2f}", color=WHITE, font_size=24)
            checkpoint_placeholder = Text("", color=WHITE, font_size=24)
            row = VGroup(policy_idx, checkpoint_placeholder, objective_value)
            row.arrange(RIGHT, buff=6)
            rows.append(row)
            checkpoint_placeholders.append(checkpoint_placeholder)

        table = VGroup(*rows)
        table.arrange(DOWN, buff=0.35)
        table.to_corner(UL, buff=1)
        table.shift(0.8 * DOWN)

        cp_move_anims = []
        for p in range(8):
            cp_move_anims.append(selected_checkpoints[p].animate.move_to(table[p].get_center() + 0.8 * RIGHT))
        self.play(*cp_move_anims, run_time=1.0)
        wait(1)

        cp_font_anims = []
        for p in range(8):
            cp_font_anims.append(selected_checkpoints[p].animate.set_font_size(24))
            cp_font_anims.append(selected_checkpoints[p].animate.set_color(WHITE))
        self.play(*cp_font_anims, run_time=0.5)

        if final_anim:
            self.play(Write(header_texts))
            self.play(Write(table))
        else:
            self.add(header_texts)
            self.add(table)

        wait(1)

        row_pos = [table[i].get_center() for i in range(8)]
        # sort objective values
        sorted_idx = list(reversed(np.argsort(objective_values)))

        # sort rows
        row_move_anims = []
        for p in range(8):
            new_row_idx = sorted_idx.index(p)
            pos = row_pos[new_row_idx]
            row_move_anims.append(table[p].animate.move_to(pos))
            row_move_anims.append(selected_checkpoints[p].animate.move_to(pos + 0.8 * RIGHT))

        if final_anim:
            self.play(*row_move_anims, run_time=1.5)
        else:
            self.play(*row_move_anims, run_time=0.001)

        wait(1)

        top_p = Rectangle(GREEN, 1.8, 8.8).move_to(row_pos[1] + 0.3 * RIGHT)
        worst_p = Rectangle(RED, 1.8, 8.8).move_to(row_pos[-2] + 0.3 * RIGHT)

        checkpoint_to_red = []
        for text in table[chosen_policy]:
            checkpoint_to_red.append(text.animate.set_color(RED))
        checkpoint_to_red.append(selected_checkpoints[chosen_policy].animate.set_color(RED))
        self.play(*checkpoint_to_red, run_time=0.5)
        self.play(Create(worst_p))
        wait(1)

        checkpoint_to_green = []
        for text in table[sorted_idx[1]]:
            checkpoint_to_green.append(text.animate.set_color(GREEN))
        checkpoint_to_green.append(selected_checkpoints[sorted_idx[1]].animate.set_color(GREEN))
        self.play(*checkpoint_to_green, run_time=0.5)
        self.play(Create(top_p))
        wait(1)

        steps = Tex(
            "\\begin{itemize}"
            "\\item Load weights and hyperparameters\\\\from a top-performing policy"
            "\\item Randomly perturb hyperparameters"
            "\\item Resume training!"
            "\\end{itemize}",
            font_size=25,
        )
        steps.to_edge(RIGHT, buff=0.2)

        self.play(Write(steps))

        self.wait(15)
