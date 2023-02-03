import copy

from manim import *


import numpy as np


def texy_text(text, **kwargs):
    """Not using Tex, just a font that looks like Tex."""
    if 'font_size' not in kwargs:
        kwargs['font_size'] = 24

    text = Tex(text, **kwargs)
    return text


def description_square(text, color):
    square = Square(side_length=0.3, color=color)
    # fill square with color
    square.set_fill(color, 1)

    group = VGroup(square, texy_text(text))
    group.arrange(RIGHT, buff=MED_SMALL_BUFF)
    return group


class SyncRL(Scene):
    def construct(self):
        title = texy_text('Synchronous Reinforcement Learning', font_size=40)

        # create three squares demonstrating that blue and orange colors correspond to simulation, inference and backpropagation
        description_squares = VGroup(
            description_square("Simulation (CPU)", BLUE_B),
            description_square("Inference (GPU)", ORANGE),
            description_square("Backpropagation (GPU)", "#ffaf75"),
        )
        description_squares.arrange(RIGHT, buff=LARGE_BUFF)

        # create lables for CPU and GPU utilization
        cpu_util_text, cpu_util_number = cpu_util_label = VGroup(
            texy_text('CPU utilization, \\%', font_size=24),
            DecimalNumber(
                0,
                show_ellipsis=False,
                num_decimal_places=1,
                include_sign=False,
                font_size=24,
                fill_color=BLUE_B,
            ),
        )
        cpu_util_label.arrange(RIGHT, buff=MED_SMALL_BUFF)

        gpu_util_text, gpu_util_number = gpu_util_label = VGroup(
            texy_text('GPU utilization, \\%', font_size=24),
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
        cpu_util_label.next_to(cpu_util_label.get_corner(DOWN+LEFT), DOWN, buff=MED_SMALL_BUFF)

        util_labels = VGroup(cpu_util_label, gpu_util_label)
        util_labels.arrange(RIGHT, buff=MED_LARGE_BUFF)

        header = VGroup(title, description_squares, util_labels)
        header.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        header.to_corner(UP + LEFT, buff=SMALL_BUFF)

        # create a manim x axis
        x_axis = NumberLine(include_tip=True, include_ticks=False, length=12)

        # move x_axis to the bottom edge of the screen
        x_axis.to_edge(DOWN).to_corner(RIGHT + DOWN)

        # add x axis tex label saying "time" to the x axis
        # x_axis_label = Text('time', font="Consolas", t2s={"time": ITALIC}, font_size=20)
        x_axis_label = texy_text('time', font_size=24)

        # position x axis label next to the tip of the x axis
        x_axis_label.next_to(x_axis.get_tip(), DOWN, SMALL_BUFF)

        num_workers = 4

        # create a vertical list with text Worker #1, Worker #2, etc.
        worker_texts = [texy_text(f'Worker {i + 1}', font_size=24) for i in range(num_workers)]
        worker_list = VGroup(*worker_texts)
        # worker_list.arrange(DOWN, buff=LARGE_BUFF)
        worker_list.arrange_in_grid(rows=4, cols=1, buff=0.5).to_edge(LEFT, SMALL_BUFF)

        # self.play(Write(worker_list))

        # move worker_list down by 0.5
        worker_list.shift(0.5 * DOWN)

        total_time_text, total_time_number = total_time_label = VGroup(
            texy_text('Total time =', font_size=24),
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
            texy_text('Observations collected ='),
            Integer(0, font_size=24),
        )
        frames_collected_label.arrange(RIGHT, buff=SMALL_BUFF)
        # frames_collected_label.next_to(total_time_label, UP, buff=MED_SMALL_BUFF)

        inference_steps_text, inference_steps_number = inference_steps_label = VGroup(
            texy_text('Inference steps ='),
            Integer(0, font_size=24),
        )
        inference_steps_label.arrange(RIGHT, buff=SMALL_BUFF)
        # inference_steps_label.next_to(frames_collected_label, UP, buff=MED_SMALL_BUFF)

        backprop_steps_text, backprop_steps_number = backprop_steps_label = VGroup(
            texy_text('Backpropagation steps ='),
            Integer(0, font_size=24),
        )
        backprop_steps_label.arrange(RIGHT, buff=SMALL_BUFF)
        # backprop_steps_label.next_to(inference_steps_label, UP, buff=MED_SMALL_BUFF)

        stats = VGroup(frames_collected_label, inference_steps_label, backprop_steps_label)
        stats.arrange(RIGHT, buff=MED_LARGE_BUFF)
        stats.next_to(total_time_label, UP, buff=MED_SMALL_BUFF)

        self.play(Write(x_axis), Write(x_axis_label), Write(worker_list), Write(total_time_label), Write(stats), Write(header))
        self.wait(1)

        initial_time = self.renderer.time
        f_always(total_time_number.set_value, lambda: self.renderer.time - initial_time)

        def timestep_animation(rect, target_w):
            curr_w = rect.width
            ratio = target_w / curr_w
            return rect.animate.stretch_about_point(ratio, 0, rect.get_left())

        step_times_arr = [
            [1, 0.5, 2.5, 1],
            [1, 3, 1, 0.5],
            [1, 0.5, 4, 1],
            [4, 1, 0.5, 3],
        ]
        scale = 0.275
        time_scale = 1.8  # debug, use 1.8
        step_times_arr = np.array(step_times_arr) * scale
        inference_time = 0.6 * scale
        backprop_time = 3.0 * scale

        stroke = 0.0
        width = 0.001
        ofs = np.array([0.05, 0, 0])

        align_with = x_axis.get_left()

        def play(animations_, run_time):
            self.play(*animations_, run_time=run_time * time_scale, rate_func=linear)

        total_frames_collected = total_inference_steps = total_backprop_steps = 0

        total_cpu_time = total_gpu_time = total_time = 0

        num_iterations = 2
        for iteration in range(num_iterations):
            for step_idx, step_times_og in enumerate(step_times_arr):
                step_times = copy.copy(step_times_og)
                env_step_rects = [Rectangle(width=width, height=0.3, stroke_width=stroke, fill_color=BLUE_B, fill_opacity=1) for i in range(num_workers)]
                for i, t in enumerate(env_step_rects):
                    t.move_to(worker_texts[i].get_center(), LEFT, coor_mask=np.array([0, 1, 0]))
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
                    total_cpu_time += len(animations) * min_step
                    cpu_util_number.set_value((100 * total_cpu_time) / (num_workers * total_time))
                    gpu_util_number.set_value(100 * total_gpu_time / total_time)

                inference_height = abs((env_step_rects[-1].get_corner(DOWN + LEFT) - env_step_rects[0].get_corner(UP + LEFT))[1])
                inference_rect = Rectangle(width=width, height=inference_height, stroke_width=stroke, fill_color=ORANGE, fill_opacity=1)

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
                total_gpu_time += inference_time
                if iteration != num_iterations - 1:
                    cpu_util_number.set_value((100 * total_cpu_time) / (num_workers * total_time))
                gpu_util_number.set_value(100 * total_gpu_time / total_time)

                if step_idx == len(step_times_arr) - 1:
                    backprop_rect = Rectangle(width=width, height=inference_height, stroke_width=stroke, fill_color="#ffaf75", fill_opacity=1)
                    backprop_rect.move_to(env_step_rects[0].get_top(), UP, coor_mask=np.array([0, 1, 0]))
                    backprop_rect.move_to(align_with + ofs, LEFT, coor_mask=np.array([1, 0, 0]))
                    self.add(backprop_rect)

                    backprop_animation = timestep_animation(backprop_rect, backprop_time)
                    play([backprop_animation], backprop_time)
                    total_time += backprop_time
                    total_gpu_time += backprop_time
                    if iteration != num_iterations - 1:
                        cpu_util_number.set_value((100 * total_cpu_time) / (num_workers * total_time))
                    gpu_util_number.set_value(100 * total_gpu_time / total_time)
                    total_backprop_steps += 1
                    backprop_steps_number.set_value(total_backprop_steps)
                    align_with = backprop_rect.get_right()

        # remove the effect of falways
        total_time_number.clear_updaters()

        self.play(Indicate(total_time_label))
        self.wait(15)


class AsyncRL(Scene):
    def construct(self):
        title = texy_text('Asynchronous Reinforcement Learning (Sample Factory)', font_size=40)

        # create three squares demonstrating that blue and orange colors correspond to simulation, inference and backpropagation
        description_squares = VGroup(
            description_square("Simulation (CPU)", BLUE_B),
            description_square("Inference (GPU)", ORANGE),
            description_square("Backpropagation (GPU)", "#ffaf75"),
        )
        description_squares.arrange(RIGHT, buff=LARGE_BUFF)

        # create lables for CPU and GPU utilization
        cpu_util_text, cpu_util_number = cpu_util_label = VGroup(
            texy_text('CPU utilization, \%', font_size=24),
            DecimalNumber(
                0,
                show_ellipsis=False,
                num_decimal_places=1,
                include_sign=False,
                font_size=24,
                fill_color=BLUE_B,
            ),
        )
        cpu_util_label.arrange(RIGHT, buff=MED_SMALL_BUFF)

        gpu_util_text, gpu_util_number = gpu_util_label = VGroup(
            texy_text('GPU utilization, \\%', font_size=24),
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
        cpu_util_label.next_to(cpu_util_label.get_corner(DOWN+LEFT), DOWN, buff=MED_SMALL_BUFF)

        util_labels = VGroup(cpu_util_label, gpu_util_label)
        util_labels.arrange(RIGHT, buff=MED_LARGE_BUFF)

        header = VGroup(title, description_squares, util_labels)
        header.arrange(DOWN, buff=MED_LARGE_BUFF * 0.8, aligned_edge=LEFT)
        header.to_corner(UP + LEFT, buff=SMALL_BUFF)

        # create a manim x axis
        x_axis = NumberLine(include_tip=True, include_ticks=False, length=11.5)

        # move x_axis to the bottom edge of the screen
        x_axis.to_edge(DOWN).to_corner(RIGHT + DOWN)

        # add x axis tex label saying "time" to the x axis
        # x_axis_label = Text('time', font="Consolas", t2s={"time": ITALIC}, font_size=20)
        x_axis_label = texy_text('time', font_size=24)

        # position x axis label next to the tip of the x axis
        x_axis_label.next_to(x_axis.get_tip(), DOWN, SMALL_BUFF)

        num_workers = 4

        # create a vertical list with text Worker #1, Worker #2, etc.
        rollout_worker_texts = [texy_text(f'Rollout worker {i + 1}', font_size=22) for i in range(num_workers)]
        policy_worker = texy_text(f'Policy worker', font_size=22)
        learner = texy_text(f'Learner', font_size=22)

        worker_list = VGroup(*(rollout_worker_texts + [policy_worker, learner]))
        worker_list.arrange_in_grid(rows=num_workers + 2, cols=1, buff=0.5).to_edge(LEFT, SMALL_BUFF)

        # move worker_list down by 0.5
        worker_list.shift(0.5 * DOWN)

        total_time_text, total_time_number = total_time_label = VGroup(
            texy_text('Total time =', font_size=24),
            DecimalNumber(
                0,
                show_ellipsis=True,
                num_decimal_places=2,
                include_sign=False,
                font_size=24,
            ),
        )
        total_time_label.arrange(RIGHT, buff=SMALL_BUFF)

        frames_collected_text, frames_collected_number = frames_collected_label = VGroup(
            texy_text('Observations collected ='),
            Integer(
                0,
                font_size=24,
            ),
        )
        frames_collected_label.arrange(RIGHT, buff=SMALL_BUFF)
        # frames_collected_label.next_to(total_time_label, UP, buff=MED_SMALL_BUFF)

        inference_steps_text, inference_steps_number = inference_steps_label = VGroup(
            texy_text('Inference steps ='),
            Integer(
                0,
                font_size=24,
            ),
        )
        inference_steps_label.arrange(RIGHT, buff=SMALL_BUFF)
        # inference_steps_label.next_to(frames_collected_label, UP, buff=MED_SMALL_BUFF)

        backprop_steps_text, backprop_steps_number = backprop_steps_label = VGroup(
            texy_text('Backpropagation steps ='),
            Integer(
                0,
                font_size=24,
            ),
        )
        backprop_steps_label.arrange(RIGHT, buff=SMALL_BUFF)
        # backprop_steps_label.next_to(inference_steps_label, UP, buff=MED_SMALL_BUFF)

        stats = VGroup(frames_collected_label, inference_steps_label, backprop_steps_label, total_time_label)
        stats.arrange(RIGHT, buff=MED_LARGE_BUFF)
        stats.next_to(x_axis, UP, buff=SMALL_BUFF)
        stats.shift(0.75 * LEFT)

        self.play(Write(x_axis), Write(x_axis_label), Write(worker_list), Write(stats), Write(header))
        self.wait(0.5)
        # self.add(x_axis, x_axis_label, worker_list, header, stats)

        initial_time = self.renderer.time
        f_always(total_time_number.set_value, lambda: self.renderer.time - initial_time)

        step_times_arr = [
            [1, 0.5, 2.5, 1, 4, 1, 0.5, 3],
            [1, 3, 1, 0.5, 1, 0.5, 4, 1],
            [1, 0.5, 4, 1, 1, 3, 1, 0.5],
            [4, 1, 0.5, 3, 1, 0.5, 2.5, 1],
        ]
        scale = 0.275
        time_scale = 1.8  # debug, use 1.5
        step_times_arr = np.array(step_times_arr) * scale
        inference_time = 0.6 * scale
        backprop_time = 3.0 * scale

        stroke = 0.0
        width = 0.001
        ofs = np.array([0.05, 0, 0])

        def play(animations_, run_time_):
            self.play(*animations_, run_time=run_time_ * time_scale, rate_func=linear)

        # repeat numpy array 2 times
        # num_iterations = 2
        # step_times_arr = np.repeat(step_times_arr, num_iterations, axis=0).reshape(4, -1)

        class WorkerState:
            def __init__(self):
                self.rect = None
                self.end_time = None

            def finish_rect(self):
                self.rect = None
                self.end_time = None

        class RolloutWorkerState(WorkerState):
            def __init__(self, work):
                super().__init__()
                self.work = list(copy.copy(work))
                self.has_actions = True
                self.trajectory_len = 0

            def init_rect(self, rect_, curr_time):
                self.rect = rect_
                self.end_time = curr_time + self.work[0]
                self.work.pop(0)
                self.has_actions = False

            def finish_rect(self):
                super().finish_rect()
                self.trajectory_len += 1

        class LearnerState(WorkerState):
            def __init__(self):
                super().__init__()
                self.num_trajectories = 0

        rollout_worker_states = [RolloutWorkerState(step_times_arr[i]) for i in range(num_workers)]
        inference_worker_state = WorkerState()
        learner_state = LearnerState()

        orig_x_coord = x_axis.get_left()[0] + ofs[0]

        def timestep_animation_2(rect_, target_right_x):
            target_right_x = orig_x_coord + target_right_x

            target_w = target_right_x - rect_.get_left()[0]
            curr_w = rect_.width
            if target_w < 0:
                target_w = curr_w

            ratio = target_w / curr_w
            return rect_.animate.stretch_about_point(ratio, 0, rect_.get_left())

        total_frames_collected = total_inference_steps = total_backprop_steps = 0
        total_cpu_time = total_gpu_time = total_time = 0

        rw_awaiting_action = []
        min_action_requests_in_queue = 2

        while True:
            active_rw = 0

            for rwi, rws in enumerate(rollout_worker_states):
                if rws.work:
                    active_rw += 1

                if rws.rect is None and rws.has_actions and rws.work:
                    # ready to process next step
                    rollout_rect = Rectangle(width=width, height=0.3, stroke_width=stroke, fill_color=BLUE_B, fill_opacity=1)
                    rollout_rect.move_to(rollout_worker_texts[rwi].get_center(), LEFT, coor_mask=np.array([0, 1, 0]))
                    rollout_rect.move_to(orig_x_coord + total_time + ofs, LEFT, coor_mask=np.array([1, 0, 0]))
                    rollout_rect.c_type__ = 'cpu'
                    rws.init_rect(rollout_rect, total_time)
                    self.add(rollout_rect)

            if rw_awaiting_action and len(rw_awaiting_action) >= min(min_action_requests_in_queue, active_rw) and inference_worker_state.rect is None:
                inference_rect = Rectangle(width=width, height=0.3, stroke_width=stroke, fill_color=ORANGE, fill_opacity=1)
                inference_rect.move_to(policy_worker.get_center(), LEFT, coor_mask=np.array([0, 1, 0]))
                inference_rect.move_to(orig_x_coord + total_time + ofs, LEFT, coor_mask=np.array([1, 0, 0]))
                inference_rect.c_type__ = 'gpu'

                inference_worker_state.rect = inference_rect
                inference_worker_state.end_time = total_time + inference_time
                inference_worker_state.rw_awaiting_action = copy.copy(rw_awaiting_action)
                rw_awaiting_action = []

                self.add(inference_rect)

            if learner_state.rect is None and learner_state.num_trajectories >= num_workers:
                learner_rect = Rectangle(width=width, height=0.3, stroke_width=stroke, fill_color="#ffaf75", fill_opacity=1)
                learner_rect.move_to(learner.get_center(), LEFT, coor_mask=np.array([0, 1, 0]))
                learner_rect.move_to(orig_x_coord + total_time + ofs, LEFT, coor_mask=np.array([1, 0, 0]))
                learner_rect.c_type__ = 'gpu'

                learner_state.num_trajectories -= num_workers
                learner_state.rect = learner_rect
                learner_state.end_time = total_time + backprop_time

                self.add(learner_rect)

            # find the shortest rect and all currently active rects
            min_end_time = 1e9
            active_rects = []
            cpu_rects = gpu_rects = 0
            # noinspection PyTypeChecker
            for rws in rollout_worker_states + [inference_worker_state, learner_state]:
                if rws.end_time is not None:
                    active_rects.append(rws.rect)
                    if rws.rect.c_type__ == 'cpu':
                        cpu_rects += 1
                    else:
                        gpu_rects = min(gpu_rects + 1, 1)
                    if rws.end_time < min_end_time:
                        min_end_time = rws.end_time

            if min_end_time == 1e9:
                break

            # update the rects
            animations = []
            for r in active_rects:
                animations.append(timestep_animation_2(r, min_end_time))

            play(animations, min_end_time - total_time)
            total_cpu_time += cpu_rects * (min_end_time - total_time)
            total_gpu_time += gpu_rects * (min_end_time - total_time)
            total_time = min_end_time

            if active_rw > 0:
                cpu_util_number.set_value((100 * total_cpu_time) / (num_workers * total_time))
            gpu_util_number.set_value((100 * total_gpu_time) / total_time)

            # check if any rects are done
            for rwi, rws in enumerate(rollout_worker_states):
                if rws.rect is not None and rws.end_time - min_end_time <= 1e-5:
                    rws.finish_rect()
                    total_frames_collected += 1
                    frames_collected_number.set_value(total_frames_collected)
                    rw_awaiting_action.append(rwi)

                    if rws.trajectory_len >= 4:
                        rws.trajectory_len -= 4
                        learner_state.num_trajectories += 1

            # check if inference is done
            if inference_worker_state.rect is not None and inference_worker_state.end_time - min_end_time <= 1e-5:
                for rwi in inference_worker_state.rw_awaiting_action:
                    rollout_worker_states[rwi].has_actions = True
                inference_worker_state.finish_rect()
                inference_worker_state.rw_awaiting_action = []
                total_inference_steps += 1
                inference_steps_number.set_value(total_inference_steps)

            # check if learning is done
            if learner_state.rect is not None and learner_state.end_time - min_end_time <= 1e-5:
                learner_state.finish_rect()
                total_backprop_steps += 1
                backprop_steps_number.set_value(total_backprop_steps)

        # remove the effect of falways
        total_time_number.clear_updaters()

        # self.play(Wiggle(util_labels))
        self.play(Indicate(total_time_label))

        self.wait(15)


class DoubleBuffered(Scene):
    def construct(self):
        title = texy_text('Double Buffered Sampling (Sample Factory)', font_size=40)

        # create three squares demonstrating that blue and orange colors correspond to simulation, inference and backpropagation
        simulation_square_1 = Square(side_length=0.3, color="#024775", stroke_width=0)
        simulation_square_1.set_fill("#024775", 1)
        simulation_square_2 = Square(side_length=0.3, color=BLUE_B, stroke_width=0)
        simulation_square_2.set_fill(BLUE_B, 1)
        simulation_square = VGroup(simulation_square_1, simulation_square_2)
        simulation_square_2.next_to(simulation_square_1.get_corner(UP + LEFT), DOWN+RIGHT, SMALL_BUFF)

        simulation_square = VGroup(simulation_square, texy_text('Simulation (CPU)'))
        simulation_square.arrange(RIGHT, buff=MED_SMALL_BUFF)

        description_squares = VGroup(
            simulation_square,
            description_square("Inference (GPU)", ORANGE),
            description_square("Backpropagation (GPU)", "#ffaf75"),
        )
        description_squares.arrange(RIGHT, buff=LARGE_BUFF)

        # create lables for CPU and GPU utilization
        cpu_util_text, cpu_util_number = cpu_util_label = VGroup(
            texy_text('CPU utilization, \%', font_size=24),
            DecimalNumber(
                0,
                show_ellipsis=False,
                num_decimal_places=1,
                include_sign=False,
                font_size=24,
                fill_color=BLUE_B,
            ),
        )
        cpu_util_label.arrange(RIGHT, buff=MED_SMALL_BUFF)

        gpu_util_text, gpu_util_number = gpu_util_label = VGroup(
            texy_text('GPU utilization, \%', font_size=24),
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
        cpu_util_label.next_to(cpu_util_label.get_corner(DOWN+LEFT), DOWN, buff=MED_SMALL_BUFF)

        util_labels = VGroup(cpu_util_label, gpu_util_label)
        util_labels.arrange(RIGHT, buff=MED_LARGE_BUFF)

        header = VGroup(title, description_squares, util_labels)
        header.arrange(DOWN, buff=MED_LARGE_BUFF * 0.8, aligned_edge=LEFT)
        header.to_corner(UP + LEFT, buff=SMALL_BUFF)

        # create a manim x axis
        x_axis = NumberLine(include_tip=True, include_ticks=False, length=11.5)

        # move x_axis to the bottom edge of the screen
        x_axis.to_edge(DOWN).to_corner(RIGHT + DOWN)

        # add x axis tex label saying "time" to the x axis
        # x_axis_label = Text('time', font="Consolas", t2s={"time": ITALIC}, font_size=20)
        x_axis_label = texy_text('time', font_size=24)

        # position x axis label next to the tip of the x axis
        x_axis_label.next_to(x_axis.get_tip(), DOWN, SMALL_BUFF)

        num_workers = 4

        # create a vertical list with text Worker #1, Worker #2, etc.
        rollout_worker_texts = [texy_text(f'Rollout worker {i + 1}', font_size=22) for i in range(num_workers)]
        policy_worker = texy_text(f'Policy worker', font_size=22)
        learner = texy_text(f'Learner', font_size=22)

        worker_list = VGroup(*(rollout_worker_texts + [policy_worker, learner]))
        worker_list.arrange_in_grid(rows=num_workers + 2, cols=1, buff=0.5).to_edge(LEFT, SMALL_BUFF)

        # move worker_list down by 0.5
        worker_list.shift(0.5 * DOWN)

        total_time_text, total_time_number = total_time_label = VGroup(
            texy_text('Total time =', font_size=24),
            DecimalNumber(
                0,
                show_ellipsis=True,
                num_decimal_places=2,
                include_sign=False,
                font_size=24,
            ),
        )
        total_time_label.arrange(RIGHT, buff=SMALL_BUFF)

        frames_collected_text, frames_collected_number = frames_collected_label = VGroup(
            texy_text('Observations collected ='),
            Integer(
                0,
                font_size=24,
            ),
        )
        frames_collected_label.arrange(RIGHT, buff=SMALL_BUFF)
        # frames_collected_label.next_to(total_time_label, UP, buff=MED_SMALL_BUFF)

        inference_steps_text, inference_steps_number = inference_steps_label = VGroup(
            texy_text('Inference steps ='),
            Integer(
                0,
                font_size=24,
            ),
        )
        inference_steps_label.arrange(RIGHT, buff=SMALL_BUFF)
        # inference_steps_label.next_to(frames_collected_label, UP, buff=MED_SMALL_BUFF)

        backprop_steps_text, backprop_steps_number = backprop_steps_label = VGroup(
            texy_text('Backpropagation steps ='),
            Integer(
                0,
                font_size=24,
            ),
        )
        backprop_steps_label.arrange(RIGHT, buff=SMALL_BUFF)
        # backprop_steps_label.next_to(inference_steps_label, UP, buff=MED_SMALL_BUFF)

        stats = VGroup(frames_collected_label, inference_steps_label, backprop_steps_label, total_time_label)
        stats.arrange(RIGHT, buff=MED_LARGE_BUFF)
        stats.next_to(x_axis, UP, buff=SMALL_BUFF)
        stats.shift(0.75 * LEFT)

        self.play(Write(x_axis), Write(x_axis_label), Write(worker_list), Write(stats), Write(header))
        self.wait(0.5)

        initial_time = self.renderer.time
        f_always(total_time_number.set_value, lambda: self.renderer.time - initial_time)

        step_times_arr = [
            [1, 0.5, 2.5, 1, 4, 1, 0.5, 3],
            [1, 3, 1, 0.5, 1, 0.5, 4, 1],
            [1, 0.5, 4, 1, 1, 3, 1, 0.5],
            [4, 1, 0.5, 3, 1, 0.5, 2.5, 1],
        ]
        scale = 0.275
        time_scale = 1.8  # debug, use 1.5
        step_times_arr = np.array(step_times_arr) * scale
        inference_time = 0.6 * scale
        backprop_time = 3.0 * scale

        stroke = 0.0
        width = 0.001
        ofs = np.array([0.05, 0, 0])

        def play(animations_, run_time_):
            self.play(*animations_, run_time=run_time_ * time_scale, rate_func=linear)

        # repeat numpy array 2 times
        # num_iterations = 2
        # step_times_arr = np.repeat(step_times_arr, num_iterations, axis=0).reshape(4, -1)

        class WorkerState:
            def __init__(self):
                self.rect = None
                self.end_time = None

            def finish_rect(self):
                self.rect = None
                self.end_time = None

        class RolloutWorkerState(WorkerState):
            def __init__(self, work):
                super().__init__()
                self.work = list(copy.copy(work))
                self.has_actions = [True, True]
                self.trajectory_len = 0

            def init_rect(self, rect_, curr_time):
                self.rect = rect_
                self.end_time = curr_time + self.work[0]
                self.work.pop(0)
                self.has_actions[self.split_idx()] = False

            def finish_rect(self):
                super().finish_rect()
                self.trajectory_len += 1

            def split_idx(self):
                split_index = self.trajectory_len % 2
                return split_index

        class LearnerState(WorkerState):
            def __init__(self):
                super().__init__()
                self.num_trajectories = 0

        rollout_worker_states = [RolloutWorkerState(step_times_arr[i]) for i in range(num_workers)]
        inference_worker_state = WorkerState()
        learner_state = LearnerState()

        orig_x_coord = x_axis.get_left()[0] + ofs[0]

        def timestep_animation_2(rect_, target_right_x):
            target_right_x = orig_x_coord + target_right_x

            target_w = target_right_x - rect_.get_left()[0]
            curr_w = rect_.width
            if target_w < 0:
                target_w = curr_w

            ratio = target_w / curr_w
            return rect_.animate.stretch_about_point(ratio, 0, rect_.get_left())

        total_frames_collected = total_inference_steps = total_backprop_steps = 0
        total_cpu_time = total_gpu_time = total_time = 0

        rw_awaiting_action = []
        min_action_requests_in_queue = 2

        while True:
            active_rw = 0

            for rwi, rws in enumerate(rollout_worker_states):
                if rws.work:
                    active_rw += 1

                if rws.rect is None and rws.has_actions[rws.split_idx()] and rws.work:
                    # ready to process next step
                    color = BLUE_B if rws.split_idx() == 0 else "#024775"

                    rollout_rect = Rectangle(width=width, height=0.3, stroke_width=stroke, fill_color=color, fill_opacity=1)
                    rollout_rect.move_to(rollout_worker_texts[rwi].get_center(), LEFT, coor_mask=np.array([0, 1, 0]))
                    rollout_rect.move_to(orig_x_coord + total_time + ofs, LEFT, coor_mask=np.array([1, 0, 0]))
                    rollout_rect.c_type__ = 'cpu'
                    rws.init_rect(rollout_rect, total_time)
                    self.add(rollout_rect)

            if rw_awaiting_action and len(rw_awaiting_action) >= min(min_action_requests_in_queue, active_rw) and inference_worker_state.rect is None:
                inference_rect = Rectangle(width=width, height=0.3, stroke_width=stroke, fill_color=ORANGE, fill_opacity=1)
                inference_rect.move_to(policy_worker.get_center(), LEFT, coor_mask=np.array([0, 1, 0]))
                inference_rect.move_to(orig_x_coord + total_time + ofs, LEFT, coor_mask=np.array([1, 0, 0]))
                inference_rect.c_type__ = 'gpu'

                inference_worker_state.rect = inference_rect
                inference_worker_state.end_time = total_time + inference_time
                inference_worker_state.rw_awaiting_action = copy.copy(rw_awaiting_action)
                rw_awaiting_action = []

                self.add(inference_rect)

            if learner_state.rect is None and learner_state.num_trajectories >= num_workers:
                learner_rect = Rectangle(width=width, height=0.3, stroke_width=stroke, fill_color="#ffaf75", fill_opacity=1)
                learner_rect.move_to(learner.get_center(), LEFT, coor_mask=np.array([0, 1, 0]))
                learner_rect.move_to(orig_x_coord + total_time + ofs, LEFT, coor_mask=np.array([1, 0, 0]))
                learner_rect.c_type__ = 'gpu'

                learner_state.num_trajectories -= num_workers
                learner_state.rect = learner_rect
                learner_state.end_time = total_time + backprop_time

                self.add(learner_rect)

            # find the shortest rect and all currently active rects
            min_end_time = 1e9
            active_rects = []
            cpu_rects = gpu_rects = 0
            # noinspection PyTypeChecker
            for rws in rollout_worker_states + [inference_worker_state, learner_state]:
                if rws.end_time is not None:
                    active_rects.append(rws.rect)
                    if rws.rect.c_type__ == 'cpu':
                        cpu_rects += 1
                    else:
                        gpu_rects = min(gpu_rects + 1, 1)
                    if rws.end_time < min_end_time:
                        min_end_time = rws.end_time

            if min_end_time == 1e9:
                break

            # update the rects
            animations = []
            for r in active_rects:
                animations.append(timestep_animation_2(r, min_end_time))

            play(animations, min_end_time - total_time)
            total_cpu_time += cpu_rects * (min_end_time - total_time)
            total_gpu_time += gpu_rects * (min_end_time - total_time)
            total_time = min_end_time

            if active_rw > 0:
                cpu_util_number.set_value((100 * total_cpu_time) / (num_workers * total_time))
            gpu_util_number.set_value((100 * total_gpu_time) / total_time)

            # check if any rects are done
            for rwi, rws in enumerate(rollout_worker_states):
                if rws.rect is not None and rws.end_time - min_end_time <= 1e-5:
                    rw_awaiting_action.append((rwi, rws.split_idx()))
                    rws.finish_rect()
                    total_frames_collected += 1
                    frames_collected_number.set_value(total_frames_collected)

                    if rws.trajectory_len >= 4:
                        rws.trajectory_len -= 4
                        learner_state.num_trajectories += 1

            # check if inference is done
            if inference_worker_state.rect is not None and inference_worker_state.end_time - min_end_time <= 1e-5:
                for rwi, split in inference_worker_state.rw_awaiting_action:
                    rollout_worker_states[rwi].has_actions[split] = True
                inference_worker_state.finish_rect()
                inference_worker_state.rw_awaiting_action = []
                total_inference_steps += 1
                inference_steps_number.set_value(total_inference_steps)

            # check if learning is done
            if learner_state.rect is not None and learner_state.end_time - min_end_time <= 1e-5:
                learner_state.finish_rect()
                total_backprop_steps += 1
                backprop_steps_number.set_value(total_backprop_steps)

        # remove the effect of falways
        total_time_number.clear_updaters()

        # self.play(Wiggle(util_labels))
        self.play(Indicate(total_time_label))

        self.wait(15)
