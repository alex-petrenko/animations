import copy
import random

import numpy as np
from manim import *


def texy_text(text, **kwargs):
    """Not using Tex, just a font that looks like Tex."""
    if "font_size" not in kwargs:
        kwargs["font_size"] = 24

    text = Tex(text, **kwargs)
    return text


def description_square(text, color):
    square = Square(side_length=0.3, color=color)
    # fill square with color
    square.set_fill(color, 1)

    group = VGroup(square, texy_text(text))
    group.arrange(RIGHT, buff=MED_SMALL_BUFF)
    return group


class SyncRL_V2(Scene):
    def construct(self):
        title = texy_text('Synchronous Reinforcement Learning "Version 2.0"', font_size=40)

        sim_color = "#f5602a"

        description_squares = VGroup(
            description_square("Simulation (GPU)", sim_color),
            description_square("Inference (GPU)", ORANGE),
            description_square("Backpropagation (GPU)", "#ffaf75"),
        )
        description_squares.arrange(RIGHT, buff=LARGE_BUFF)

        gpu_util_text, gpu_util_number = gpu_util_label = VGroup(
            texy_text("GPU utilization, \\%", font_size=24),
            DecimalNumber(
                0,
                show_ellipsis=False,
                num_decimal_places=1,
                include_sign=False,
                font_size=24,
                fill_color=ORANGE,
            ),
        )
        gpu_util_label.arrange(RIGHT, buff=MED_SMALL_BUFF)

        header = VGroup(title, description_squares, gpu_util_label)
        header.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        header.to_corner(UP + LEFT, buff=SMALL_BUFF)

        gpu_util_label.shift(RIGHT)

        # create a manim x axis
        x_axis = NumberLine(include_tip=True, include_ticks=False, length=13)

        # move x_axis to the bottom edge of the screen
        x_axis.to_edge(DOWN).to_corner(RIGHT + DOWN)

        # add x axis tex label saying "time" to the x axis
        # x_axis_label = Text('time', font="Consolas", t2s={"time": ITALIC}, font_size=20)
        x_axis_label = texy_text("time", font_size=24)

        # position x axis label next to the tip of the x axis
        x_axis_label.next_to(x_axis.get_tip(), DOWN, SMALL_BUFF)

        debug = False

        num_workers = 21 if debug else 21

        total_time_text, total_time_number = total_time_label = VGroup(
            texy_text("Total time =", font_size=24),
            DecimalNumber(
                0,
                show_ellipsis=True,
                num_decimal_places=2,
                include_sign=False,
                font_size=24,
            ),
        )
        total_time_label.arrange(RIGHT, buff=SMALL_BUFF)
        total_time_label.next_to(x_axis, UP, buff=SMALL_BUFF)

        frames_collected_text, frames_collected_number = frames_collected_label = VGroup(
            texy_text("Observations collected ="),
            Integer(0, font_size=24),
        )
        frames_collected_label.arrange(RIGHT, buff=SMALL_BUFF)
        # frames_collected_label.next_to(total_time_label, UP, buff=MED_SMALL_BUFF)

        inference_steps_text, inference_steps_number = inference_steps_label = VGroup(
            texy_text("Inference steps ="),
            Integer(0, font_size=24),
        )
        inference_steps_label.arrange(RIGHT, buff=SMALL_BUFF)
        # inference_steps_label.next_to(frames_collected_label, UP, buff=MED_SMALL_BUFF)

        backprop_steps_text, backprop_steps_number = backprop_steps_label = VGroup(
            texy_text("Backpropagation steps ="),
            Integer(0, font_size=24),
        )
        backprop_steps_label.arrange(RIGHT, buff=SMALL_BUFF)
        # backprop_steps_label.next_to(inference_steps_label, UP, buff=MED_SMALL_BUFF)

        stats = VGroup(frames_collected_label, inference_steps_label, backprop_steps_label)
        stats.arrange(RIGHT, buff=MED_LARGE_BUFF)
        stats.next_to(total_time_label, UP, buff=MED_SMALL_BUFF)

        if not debug:
            self.play(Write(x_axis), Write(x_axis_label), Write(total_time_label), Write(stats), Write(header))
            self.wait(0.5)
        else:
            self.add(x_axis, x_axis_label, total_time_label, stats, header)

        initial_time = self.renderer.time
        f_always(total_time_number.set_value, lambda: self.renderer.time - initial_time)

        def timestep_animation(rect, target_w):
            curr_w = rect.width
            ratio = target_w / curr_w
            return rect.animate.stretch_about_point(ratio, 0, rect.get_left())

        step_times_arr = []

        rollout = 8

        step_times = [1, 0.9, 1.1, 1.05, 0.95]
        for i in range(rollout):
            # sample random number from step_times:
            step_times_arr.append([random.choice(step_times) for _ in range(num_workers)])

        scale = 0.29
        time_scale = 0.3 if debug else 1.1  # debug, use 1.8
        step_times_arr = np.array(step_times_arr) * scale
        inference_time = 0.7 * scale
        backprop_time = 3.5 * scale

        stroke = 0.0
        width = 0.001
        ofs = np.array([0.05, 0, 0])

        align_with = x_axis.get_left()

        def play(animations_, run_time):
            self.play(*animations_, run_time=run_time * time_scale, rate_func=linear)

        total_frames_collected = total_inference_steps = total_backprop_steps = 0

        total_gpu_time = total_time = 0

        num_iterations = 2 if debug else 2
        for iteration in range(num_iterations):
            for step_idx, step_times_og in enumerate(step_times_arr):
                step_times = copy.copy(step_times_og)
                env_step_rects = [
                    Rectangle(width=width, height=0.15, stroke_width=stroke, fill_color=sim_color, fill_opacity=1)
                    for i in range(num_workers)
                ]
                for i, t in enumerate(env_step_rects):
                    t.align_to(gpu_util_label.get_center(), UP)

                    # move down i times rectangle height
                    t.shift((i + 2) * t.height * 1.2 * DOWN)

                    # align t horizontally with the align_with + ofs point
                    t.align_to(align_with + ofs, LEFT)

                    # t.move_to(align_with + ofs, LEFT, coor_mask=np.array([0, 1, 0]))
                    self.add(t)

                time_so_far = 0
                while True:
                    # find the smallest non-zero element in step_times
                    min_step = 1e9
                    for step in step_times:
                        if 0 < step < min_step:
                            min_step = step

                    if min_step == 1e9:
                        break

                    animations = []
                    for i in range(num_workers):
                        if step_times[i] > 0:
                            animations.append(timestep_animation(env_step_rects[i], time_so_far + min_step))

                    # subtract curr_step_time from all remaining step times
                    for i in range(len(step_times)):
                        if step_times[i] > 0:
                            step_times[i] -= min_step
                            if step_times[i] <= 0:
                                total_frames_collected += 1
                                frames_collected_number.set_value(total_frames_collected)

                    play(animations, min_step)

                    time_so_far += min_step
                    total_time += min_step
                    total_gpu_time += len(animations) * min_step
                    # cpu_util_number.set_value((100 * total_cpu_time) / (num_workers * total_time))
                    gpu_util_number.set_value(100 * total_gpu_time / (num_workers * total_time))

                inference_height = abs(
                    (env_step_rects[-1].get_corner(DOWN + LEFT) - env_step_rects[0].get_corner(UP + LEFT))[1]
                )
                inference_rect = Rectangle(
                    width=width, height=inference_height, stroke_width=stroke, fill_color=ORANGE, fill_opacity=1
                )

                max_x_coord = np.array([-1e9, 0, 0])
                for r in env_step_rects:
                    if r.get_right()[0] > max_x_coord[0]:
                        max_x_coord = r.get_right()
                align_with = max_x_coord
                inference_rect.move_to(env_step_rects[0].get_top(), UP, coor_mask=np.array([0, 1, 0]))
                inference_rect.move_to(align_with + ofs, LEFT, coor_mask=np.array([1, 0, 0]))

                self.add(inference_rect)

                inference_animation = timestep_animation(inference_rect, inference_time)

                play([inference_animation], inference_time)
                total_inference_steps += 1
                inference_steps_number.set_value(total_inference_steps)
                align_with = inference_rect.get_right()
                total_time += inference_time
                total_gpu_time += inference_time * num_workers
                # if iteration != num_iterations - 1:
                #     cpu_util_number.set_value((100 * total_cpu_time) / (num_workers * total_time))
                gpu_util_number.set_value(100 * total_gpu_time / (num_workers * total_time))

                if step_idx == len(step_times_arr) - 1:
                    backprop_rect = Rectangle(
                        width=width, height=inference_height, stroke_width=stroke, fill_color="#ffaf75", fill_opacity=1
                    )
                    backprop_rect.move_to(env_step_rects[0].get_top(), UP, coor_mask=np.array([0, 1, 0]))
                    backprop_rect.move_to(align_with + ofs, LEFT, coor_mask=np.array([1, 0, 0]))
                    self.add(backprop_rect)

                    backprop_animation = timestep_animation(backprop_rect, backprop_time)
                    play([backprop_animation], backprop_time)
                    total_time += backprop_time
                    total_gpu_time += backprop_time * num_workers
                    # if iteration != num_iterations - 1:
                    #     cpu_util_number.set_value((100 * total_cpu_time) / (num_workers * total_time))
                    gpu_util_number.set_value(100 * total_gpu_time / (num_workers * total_time))
                    total_backprop_steps += 1
                    backprop_steps_number.set_value(total_backprop_steps)
                    align_with = backprop_rect.get_right()

        # remove the effect of falways
        total_time_number.clear_updaters()

        if not debug:
            self.wait(1.0)
            self.play(Indicate(gpu_util_label))
            self.wait(30)
