import copy
import random

from manim import *


class PBT(Scene):
    # noinspection PyTypeChecker
    def construct(self):
        """
        Animation explaining the mechanism of population based training.
        """

        random.seed(42)
        random.random()
        rng_state = random.getstate()

        final_anim = True

        def _write(text_, **kwargs):
            if final_anim:
                self.play(Write(text_, **kwargs))
            else:
                self.add(text_)

        def _wait(t):
            if final_anim:
                self.wait(t)

        # add title
        title = Tex("Decentralized Population-Based Training", font_size=60)
        title.to_edge(UP)
        _write(title)
        _wait(0.5)

        # add subtitle
        shared_folders = Tex("Shared folders with policy checkpoints...", font_size=40)
        training = Tex("Training...", font_size=40)
        subtitles = VGroup(shared_folders, training)
        subtitles.arrange(RIGHT, buff=1)

        subtitles.next_to(title, DOWN, buff=0.35)
        _write(shared_folders)
        _wait(0.5)

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
                box.shift(i * 2.5 * DOWN + j * 3.65 * RIGHT - 5.5 * RIGHT + 0.2 * UP)
                text = Tex(f"/shared/pbt/policy{i * 4 + j}", color=WHITE, font_size=30)
                text.next_to(box, UP)  # Position text above box

                boxes.append(box)
                fnames.append(text)

        if final_anim:
            self.play(*[Create(box) for box in boxes], *[Write(text) for text in fnames])
        else:
            self.add(*boxes, *fnames)
        _wait(0.1)

        _write(training)
        _wait(0.5)

        # simulate how new checkpoints are added to each folder in the population

        delays = [0, 1, 1, 0, 1, 1, 2, 0]
        chosen_policy = 5
        delays[chosen_policy] = 1

        random.setstate(rng_state)
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
                fitness_value = fitness[p]
                text = Tex(f"p{p}-iter{this_iter:02d}-obj{fitness_value:.2f}.pth", color=WHITE, font_size=20)
                text.fitness_value = fitness_value
                text.next_to(boxes[p], DOWN, buff=-0.6 + this_iter * 0.35, aligned_edge=UP)
                iter_checkpoints.append(text)
                checkpoints[p].append(text)

            # add the new checkpoints to the scene
            if final_anim:
                self.play(*[Write(text) for text in iter_checkpoints])
            else:
                self.add(*iter_checkpoints)
            _wait(0.5)

        # highlight animation for the chosen policy
        if final_anim:
            self.play(boxes[chosen_policy].animate.set_color(YELLOW), run_time=0.5)
            self.play(Indicate(checkpoints[chosen_policy][-1], scale_factor=1.1))
        _wait(1)

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

        _wait(1)

        checkpoint_fadeouts = []
        for p in range(8):
            for checkpoint in checkpoints[p]:
                if checkpoint not in selected_checkpoints:
                    checkpoint_fadeouts.append(checkpoint)

        if final_anim:
            self.play(FadeOut(subtitles))
            self.play(*[*[FadeOut(b) for b in boxes], *[FadeOut(f) for f in fnames]])
            self.play(*[FadeOut(c) for c in checkpoint_fadeouts])
        else:
            self.remove(shared_folders)
            self.remove(training)
            self.remove(subtitles)
            self.remove(*fnames)
            self.remove(*boxes)
            self.remove(*checkpoint_fadeouts)

        headers = ["Policy index", "Checkpoint", "Objective (fitness)"]
        headers = [Tex(h, color=WHITE, font_size=30) for h in headers]
        header_texts = VGroup(*headers)
        header_texts.arrange(RIGHT, buff=1.6)
        header_texts.to_corner(UL, buff=1)
        header_texts.shift(0.4 * DOWN)

        # add the table
        rows = []
        checkpoint_placeholders = []
        objective_values = []
        for p in range(8):
            policy_idx = Tex(f"Policy \\#{p}", color=WHITE, font_size=24)
            objective_value: float = selected_checkpoints[p].fitness_value
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
        table.shift(1.2 * DOWN)

        cp_move_anims = []
        for p in range(8):
            cp_move_anims.append(selected_checkpoints[p].animate.move_to(table[p].get_center() + 0.8 * RIGHT))
        self.play(*cp_move_anims, run_time=1.0)
        _wait(1)

        cp_font_anims = []
        for p in range(8):
            font_size = 24 if p == chosen_policy else 24
            cp_font_anims.append(selected_checkpoints[p].animate.set_font_size(font_size))
        self.play(*cp_font_anims, run_time=0.5)
        cp_font_anims = []
        for p in range(8):
            color_ = YELLOW if p == chosen_policy else WHITE
            cp_font_anims.append(selected_checkpoints[p].animate.set_color(color_))
        self.play(*cp_font_anims, run_time=0.5)

        _write(header_texts)
        _write(table)
        _wait(1)

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

        _wait(1)

        top_p = Rectangle(GREEN, 1.8, 8.8).move_to(row_pos[1] + 0.3 * RIGHT)
        worst_p = Rectangle(RED, 1.8, 8.8).move_to(row_pos[-2] + 0.3 * RIGHT)

        checkpoint_to_red = []
        for text in table[chosen_policy]:
            checkpoint_to_red.append(text.animate.set_color(RED))
        checkpoint_to_red.append(selected_checkpoints[chosen_policy].animate.set_color(RED))
        if final_anim:
            self.play(*checkpoint_to_red, run_time=0.5)
            self.play(Create(worst_p))
        else:
            self.play(*checkpoint_to_red, run_time=0.001)
            self.add(worst_p)
        _wait(1)

        checkpoint_to_green = []
        for text in table[sorted_idx[1]]:
            checkpoint_to_green.append(text.animate.set_color(GREEN))
        checkpoint_to_green.append(selected_checkpoints[sorted_idx[1]].animate.set_color(GREEN))
        if final_anim:
            self.play(*checkpoint_to_green, run_time=0.5)
            self.play(Create(top_p))
        else:
            self.play(*checkpoint_to_green, run_time=0.001)
            self.add(top_p)
        _wait(1)

        steps = Tex(
            "\\begin{itemize}"
            "\\item Load weights, hyperparameters,\\\\and reward coefficients\\\\from a top-performing policy"
            "\\item Randomly perturb hyperparameters\\\\and shaping coefficients"
            "\\end{itemize}",
            font_size=25,
        )
        steps.to_edge(RIGHT, buff=0.2)

        _write(steps)

        # fade out all checkpoints except the top and worst
        fadeout_anims = []
        for p in range(8):
            if p != sorted_idx[1] and p != chosen_policy:
                fadeout_anims.append(FadeOut(table[p]))
                fadeout_anims.append(FadeOut(selected_checkpoints[p]))

        self.play(*fadeout_anims)

        # animate weights, hyperparameters, and reward coefficients being loaded from the top policy
        weights = Tex("policy weights", font_size=28, color=GREEN)
        weights.move_to(selected_checkpoints[sorted_idx[1]].get_center() + 0.4 * DOWN)

        self.play(FadeIn(weights))
        _wait(0.5)
        self.play(weights.animate.move_to(selected_checkpoints[chosen_policy].get_center() + 0.4 * UP), run_time=1.7)
        self.play(FadeOut(weights), run_time=1.5)
        del weights

        hparams = Tex("perturb(hyperparameters)", font_size=28, color=GREEN)
        hparams.move_to(selected_checkpoints[sorted_idx[1]].get_center() + 0.3 * DOWN)

        self.play(FadeIn(hparams))
        _wait(0.5)
        self.play(hparams.animate.move_to(selected_checkpoints[chosen_policy].get_center() + 0.4 * UP), run_time=1.7)
        self.play(FadeOut(hparams), run_time=1.5)
        del hparams

        coeffs = Tex("perturb(reward shaping coefficients)", font_size=28, color=GREEN)
        coeffs.move_to(selected_checkpoints[sorted_idx[1]].get_center() + 0.3 * DOWN)

        self.play(FadeIn(coeffs))
        _wait(0.5)
        self.play(coeffs.animate.move_to(selected_checkpoints[chosen_policy].get_center() + 0.4 * UP), run_time=1.7)
        self.play(FadeOut(coeffs), run_time=1.5)
        del coeffs

        self.play(
            FadeOut(top_p),
            FadeOut(worst_p),
            FadeOut(selected_checkpoints[sorted_idx[1]]),
            FadeOut(table[sorted_idx[1]]),
        )

        # Resume training!
        cp_to_green = []
        for text in table[chosen_policy]:
            cp_to_green.append(text.animate.set_color(GREEN))
        cp_to_green.append(selected_checkpoints[chosen_policy].animate.set_color(GREEN))
        self.play(*cp_to_green)

        cp_to_green = []
        for text in table[chosen_policy]:
            cp_to_green.append(text.animate.scale(1.3))
        cp_to_green.append(selected_checkpoints[chosen_policy].animate.scale(1.3))
        self.play(*cp_to_green)

        cp_to_green = []
        for text in table[chosen_policy]:
            cp_to_green.append(text.animate.shift(2 * UP))
        cp_to_green.append(selected_checkpoints[chosen_policy].animate.shift(2 * UP))
        self.play(*cp_to_green)

        resume = Tex("Resume training\\\\with updated weights!", font_size=36, color=GREEN, tex_environment="flushleft")
        resume.move_to(steps.get_bottom() + 0.9 * DOWN + 0.6 * LEFT)
        _write(resume)

        self.wait(15)
