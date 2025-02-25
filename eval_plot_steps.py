"""
VM: cpu, cpu, mem, mem, cpu % 16, cpu % 16  (0 is full, 1 is empty)
PM: cpu, cpu, mem, mem, fragment_rate, cpu % 16, fragment_rate, cpu % 16
cpu % 16 = round(normalized_cpu * 88) % 16 / 16
fragment_rate = round(normalized_cpu * 88) % 16 / round(normalized_cpu * 88)
To rescale memory, mem * 368776
"""

import argparse
import os
import random
from distutils.util import strtobool
from collections import defaultdict

import pandas as pd
import wandb
from copy import deepcopy
import cv2
from PIL import Image, ImageDraw

import gym
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import gym_reschdule_combination.envs.vm_rescheduler_env

import models
import utils
from env_patch import AsyncVectorEnv_Patch
from main import make_env, Agent
from eval_sparse import parse_args


color_dict = {
    'large': (252, 246, 174),
    'xlarge': (181, 237, 159),
    '2xlarge': (159, 237, 218),
    '4xlarge': (159, 194, 237),
    '8xlarge': (208, 159, 237),
    '16xlarge': (194, 4, 149),
    '22xlarge': (201, 119, 113),
    'free': (168, 164, 163)
}
type_list = ['large', 'xlarge', '2xlarge', '4xlarge', '8xlarge', '16xlarge', '22xlarge', 'free']
pix_per_cpu = 308 // 44
numa_height = 308
numa_width = 20
numa_gap = 5
pm_gap = 15
side_gap = 50
middle_gap = 100
legend_gap = 150
pm_width = numa_width * 2 + numa_gap
middle_y = middle_gap // 2 + numa_height + side_gap


def image_background(nml, pm_index):
    assert nml % 2 == 0, 'Only double nml is supported'

    w = (pm_width + pm_gap) * nml // 2 - pm_gap + 2 * 20
    h = numa_height * 3 + 2 * side_gap + 2 * middle_gap
    im = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for i in range(2):
        for j in range(nml // 2):
            draw.rectangle([j * (pm_width + pm_gap) + 20,
                            i * (numa_height + middle_gap) + side_gap,
                            j * (pm_width + pm_gap) + numa_width + 20,
                            i * (numa_height + middle_gap) + numa_height + side_gap],
                           fill=(141, 219, 252),
                           outline='black')
            draw.rectangle([j * (pm_width + pm_gap) + numa_width + numa_gap + 20,
                            i * (numa_height + middle_gap) + side_gap,
                            j * (pm_width + pm_gap) + pm_width + 20,
                            i * (numa_height + middle_gap) + numa_height + side_gap],
                           fill=(141, 219, 252),
                           outline='black')

    for i, vm_type in enumerate(type_list):
        draw.rectangle([i * legend_gap + 10,
                        h - 15,
                        i * legend_gap + 20,
                        h - 5],
                       fill=color_dict[vm_type],
                       outline='black')
    im_np = np.array(im)

    for i, vm_type in enumerate(type_list):
        cv2.putText(im_np, vm_type,
                    (i * legend_gap + 30, h - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))

    for j in range(nml // 2):
        cv2.putText(im_np, f"PM {int(pm_index[j]):03}",
                    (j * (pm_width + pm_gap) + 20 + numa_width // 4, side_gap - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
    for j in range(nml // 2):
        cv2.putText(im_np, f"PM {int(pm_index[nml // 2 + j]):03}",
                    (j * (pm_width + pm_gap) + 20 + numa_width // 4, 2 * numa_height + middle_gap + side_gap + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
    return im_np


def plot_pm_details(frag_rate, all_rewards, pm_cpu_details, pm_involved, vm_type_step, max_steps, num_pm, run_name,
                    restore_name, file_index):
    num_envs = pm_involved.shape[0]
    num_trials = pm_involved.shape[1] // max_steps
    for env_id in range(num_envs):
        for trial_id in range(num_trials):
            current_fr = frag_rate[env_id, max_steps * trial_id:max_steps * (trial_id + 1)]
            current_rewards = all_rewards[env_id, max_steps * trial_id:max_steps * (trial_id + 1)]
            current_pm_cpu = pm_cpu_details[env_id, max_steps * trial_id:max_steps * (trial_id + 1)]
            current_pm_involved = pm_involved[env_id, max_steps * trial_id:max_steps * (trial_id + 1)]
            current_vm_types = vm_type_step[env_id, max_steps * trial_id:max_steps * (trial_id + 1)]
            involved_index = [*set(current_pm_involved[:, :2].reshape(-1).tolist())]
            if len(involved_index) % 2 != 0:  # to make sure num_pm is even
                involved_index.append((set(range(num_pm)) - set(involved_index)).pop())
            involved_index.sort()
            involved_steps = defaultdict(list)
            for step in range(max_steps):
                involved_steps[current_pm_involved[step, 0]].append(step)
                if current_pm_involved[step, 0] != current_pm_involved[step, 1]:
                    involved_steps[current_pm_involved[step, 1]].append(step)

            current_pm_cpu = current_pm_cpu[:, np.array(involved_index, dtype=np.int32)]
            num_involved = len(involved_index)
            pm_to_idx = {k: v for v, k in enumerate(involved_index)}
            background = image_background(num_involved, involved_index)
            frames = [Image.fromarray(np.zeros_like(background))]

            for step in range(max_steps):
                img = deepcopy(background)
                source_pm = pm_to_idx[current_pm_involved[step, 0]]
                dest_pm = pm_to_idx[current_pm_involved[step, 1]]
                source_x0 = (source_pm % (num_involved // 2)) * (pm_width + pm_gap) + 20 + numa_width // 2
                source_x1 = (source_pm % (num_involved // 2)) * (pm_width + pm_gap) + 20 + \
                            numa_width + numa_gap + numa_width // 2
                dest_x0 = (dest_pm % (num_involved // 2)) * (pm_width + pm_gap) + 20 + numa_width // 2
                dest_x1 = (dest_pm % (num_involved // 2)) * (pm_width + pm_gap) + 20 + \
                          numa_width + numa_gap + numa_width // 2
                source_y = middle_y + (-1) ** (source_pm <= num_involved // 2) * middle_gap // 2
                dest_y = middle_y + (-1) ** (dest_pm <= num_involved // 2) * middle_gap // 2
                if current_pm_involved[step, 2] == 2:
                    img = cv2.line(img, (source_x0, source_y), (source_x0, middle_y), (246, 123, 210), 2)
                    img = cv2.line(img, (source_x1, source_y), (source_x1, middle_y), (246, 123, 210), 2)
                elif current_pm_involved[step, 2] == 0:
                    img = cv2.line(img, (source_x0, source_y), (source_x0, middle_y), (246, 123, 210), 2)
                elif current_pm_involved[step, 2] == 1:
                    img = cv2.line(img, (source_x1, source_y), (source_x1, middle_y), (246, 123, 210), 2)
                if current_pm_involved[step, 3] == 2:
                    img = cv2.arrowedLine(img, (dest_x0, middle_y), (dest_x0, dest_y), (246, 123, 210), 2)
                    img = cv2.arrowedLine(img, (dest_x1, middle_y), (dest_x1, dest_y), (246, 123, 210), 2)
                elif current_pm_involved[step, 3] == 0:
                    img = cv2.arrowedLine(img, (dest_x0, middle_y), (dest_x0, dest_y), (246, 123, 210), 2)
                elif current_pm_involved[step, 3] == 1:
                    img = cv2.arrowedLine(img, (dest_x1, middle_y), (dest_x1, dest_y), (246, 123, 210), 2)

                if source_x0 < dest_x1:
                    horizon_line_start = source_x1 if current_pm_involved[step, 2] == 1 else source_x0
                    horizon_line_end = dest_x0 if current_pm_involved[step, 3] == 0 else dest_x1
                else:
                    horizon_line_start = source_x0 if current_pm_involved[step, 2] == 0 else source_x1
                    horizon_line_end = dest_x1 if current_pm_involved[step, 3] == 1 else dest_x0
                img = cv2.line(img, (horizon_line_start, middle_y), (horizon_line_end, middle_y), (246, 123, 210), 2)

                img = Image.fromarray(img, 'RGB')
                draw = ImageDraw.Draw(img)
                for j in range(num_involved // 2):
                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        draw.rectangle([j * (pm_width + pm_gap) + 20,
                                        side_gap + current_cpu,
                                        j * (pm_width + pm_gap) + numa_width + 20,
                                        side_gap + current_cpu + current_pm_cpu[step, j, 0, k] * pix_per_cpu],
                                       fill=color_dict[vm_type],
                                       outline='black')
                        current_cpu += current_pm_cpu[step, j, 0, k] * pix_per_cpu
                    assert current_cpu == 44 * pix_per_cpu

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        draw.rectangle([j * (pm_width + pm_gap) + numa_width + numa_gap + 20,
                                        side_gap + current_cpu,
                                        j * (pm_width + pm_gap) + 2 * numa_width + numa_gap + 20,
                                        side_gap + current_cpu + current_pm_cpu[step, j, 1, k] * pix_per_cpu],
                                       fill=color_dict[vm_type],
                                       outline='black')
                        current_cpu += current_pm_cpu[step, j, 1, k] * pix_per_cpu
                    assert current_cpu == 44 * pix_per_cpu

                for j in range(num_involved // 2):
                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        draw.rectangle([j * (pm_width + pm_gap) + 20,
                                        side_gap + current_cpu + middle_gap + numa_height,
                                        j * (pm_width + pm_gap) + numa_width + 20,
                                        side_gap + current_cpu + current_pm_cpu[step, j + num_involved // 2, 0, k] * \
                                        pix_per_cpu + middle_gap + numa_height],
                                       fill=color_dict[vm_type],
                                       outline='black')
                        current_cpu += current_pm_cpu[step, j + num_involved // 2, 0, k] * pix_per_cpu
                    assert current_cpu == 44 * pix_per_cpu

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        draw.rectangle([j * (pm_width + pm_gap) + numa_width + numa_gap + 20,
                                        side_gap + current_cpu + middle_gap + numa_height,
                                        j * (pm_width + pm_gap) + 2 * numa_width + numa_gap + 20,
                                        side_gap + current_cpu + current_pm_cpu[step, j + num_involved // 2, 1, k] * \
                                        pix_per_cpu + middle_gap + numa_height],
                                       fill=color_dict[vm_type],
                                       outline='black')
                        current_cpu += current_pm_cpu[step, j + num_involved // 2, 1, k] * pix_per_cpu
                    assert current_cpu == 44 * pix_per_cpu

                # selected PMs
                last_y_start = side_gap + 2 * (numa_height + middle_gap)
                current_cpu = 0
                for k, vm_type in enumerate(type_list):
                    draw.rectangle([3 * (pm_width + pm_gap) + 20, last_y_start + current_cpu,
                                    3 * (pm_width + pm_gap) + numa_width + 20,
                                    last_y_start + current_cpu + current_pm_cpu[step, source_pm, 0, k] * pix_per_cpu],
                                   fill=color_dict[vm_type],
                                   outline='black')
                    current_cpu += current_pm_cpu[step, source_pm, 0, k] * pix_per_cpu

                current_cpu = 0
                for k, vm_type in enumerate(type_list):
                    draw.rectangle([3 * (pm_width + pm_gap) + numa_width + numa_gap + 20, last_y_start + current_cpu,
                                    3 * (pm_width + pm_gap) + 2 * numa_width + numa_gap + 20,
                                    last_y_start + current_cpu + current_pm_cpu[step, source_pm, 1, k] * pix_per_cpu],
                                   fill=color_dict[vm_type],
                                   outline='black')
                    current_cpu += current_pm_cpu[step, source_pm, 1, k] * pix_per_cpu

                current_cpu = 0
                for k, vm_type in enumerate(type_list):
                    draw.rectangle([4 * (pm_width + pm_gap) + 20, last_y_start + current_cpu,
                                    4 * (pm_width + pm_gap) + numa_width + 20,
                                    last_y_start + current_cpu + current_pm_cpu[step, dest_pm, 0, k] * pix_per_cpu],
                                   fill=color_dict[vm_type],
                                   outline='black')
                    current_cpu += current_pm_cpu[step, dest_pm, 0, k] * pix_per_cpu

                current_cpu = 0
                for k, vm_type in enumerate(type_list):
                    draw.rectangle([4 * (pm_width + pm_gap) + numa_width + numa_gap + 20,
                                    last_y_start + current_cpu,
                                    4 * (pm_width + pm_gap) + 2 * numa_width + numa_gap + 20,
                                    last_y_start + current_cpu + current_pm_cpu[step, dest_pm, 1, k] * pix_per_cpu],
                                   fill=color_dict[vm_type],
                                   outline='black')
                    current_cpu += current_pm_cpu[step, dest_pm, 1, k] * pix_per_cpu

                if step != 0:
                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        draw.rectangle([20, last_y_start + current_cpu, numa_width + 20,
                                        last_y_start + current_cpu + current_pm_cpu[
                                            step - 1, source_pm, 0, k] * pix_per_cpu],
                                       fill=color_dict[vm_type],
                                       outline='black')
                        current_cpu += current_pm_cpu[step - 1, source_pm, 0, k] * pix_per_cpu

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        draw.rectangle([numa_width + numa_gap + 20, last_y_start + current_cpu,
                                        2 * numa_width + numa_gap + 20,
                                        last_y_start + current_cpu + current_pm_cpu[
                                            step - 1, source_pm, 1, k] * pix_per_cpu],
                                       fill=color_dict[vm_type],
                                       outline='black')
                        current_cpu += current_pm_cpu[step - 1, source_pm, 1, k] * pix_per_cpu

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        draw.rectangle([pm_width + pm_gap + 20, last_y_start + current_cpu,
                                        pm_width + pm_gap + numa_width + 20,
                                        last_y_start + current_cpu + current_pm_cpu[
                                            step - 1, dest_pm, 0, k] * pix_per_cpu],
                                       fill=color_dict[vm_type],
                                       outline='black')
                        current_cpu += current_pm_cpu[step - 1, dest_pm, 0, k] * pix_per_cpu

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        draw.rectangle([pm_width + pm_gap + numa_width + numa_gap + 20,
                                        last_y_start + current_cpu,
                                        pm_width + pm_gap + 2 * numa_width + numa_gap + 20,
                                        last_y_start + current_cpu + current_pm_cpu[
                                            step - 1, dest_pm, 1, k] * pix_per_cpu],
                                       fill=color_dict[vm_type],
                                       outline='black')
                        current_cpu += current_pm_cpu[step - 1, dest_pm, 1, k] * pix_per_cpu

                x = np.array(img)
                for j in range(num_involved // 2):
                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        if current_pm_cpu[step, j, 0, k] != 0:
                            cv2.putText(x, f"{int(current_pm_cpu[step, j, 0, k]):02}",
                                        (j * (pm_width + pm_gap) + 20 + numa_width // 8, side_gap + current_cpu + 10),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                            current_cpu += int(current_pm_cpu[step, j, 0, k] * pix_per_cpu)
                    assert current_cpu == 44 * pix_per_cpu

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        if current_pm_cpu[step, j, 1, k] != 0:
                            cv2.putText(x, f"{int(current_pm_cpu[step, j, 1, k]):02}",
                                        (j * (pm_width + pm_gap) + 20 + numa_width + numa_width // 8 + numa_gap,
                                         side_gap + current_cpu + 10),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                            current_cpu += int(current_pm_cpu[step, j, 1, k] * pix_per_cpu)
                    assert current_cpu == 44 * pix_per_cpu

                for j in range(num_involved // 2):
                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        if current_pm_cpu[step, j + num_involved // 2, 0, k] != 0:
                            cv2.putText(x, f"{int(current_pm_cpu[step, j + num_involved // 2, 0, k]):02}",
                                        (j * (pm_width + pm_gap) + 20 + numa_width // 8,
                                         side_gap + current_cpu + 10 + middle_gap + numa_height),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                            current_cpu += int(current_pm_cpu[step, j + num_involved // 2, 0, k] * pix_per_cpu)
                    assert current_cpu == 44 * pix_per_cpu

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        if current_pm_cpu[step, j + num_involved // 2, 1, k] != 0:
                            cv2.putText(x, f"{int(current_pm_cpu[step, j + num_involved // 2, 1, k]):02}",
                                        (j * (pm_width + pm_gap) + 20 + numa_width + numa_width // 8 + numa_gap,
                                         side_gap + current_cpu + 10 + middle_gap + numa_height),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                            current_cpu += int(current_pm_cpu[step, j + num_involved // 2, 1, k] * pix_per_cpu)
                    assert current_cpu == 44 * pix_per_cpu

                # selected PMs
                current_cpu = 0
                for k, vm_type in enumerate(type_list):
                    if current_pm_cpu[step, source_pm, 0, k] != 0:
                        cv2.putText(x, f"{int(current_pm_cpu[step, source_pm, 0, k]):02}",
                                    (3 * (pm_width + pm_gap) + 20 + numa_width // 8, last_y_start + current_cpu + 10),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                        current_cpu += int(current_pm_cpu[step, source_pm, 0, k] * pix_per_cpu)

                current_cpu = 0
                for k, vm_type in enumerate(type_list):
                    if current_pm_cpu[step, source_pm, 1, k] != 0:
                        cv2.putText(x, f"{int(current_pm_cpu[step, source_pm, 1, k]):02}",
                                    (3 * (pm_width + pm_gap) + 20 + numa_width + numa_width // 8 + numa_gap,
                                     last_y_start + current_cpu + 10),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                        current_cpu += int(current_pm_cpu[step, source_pm, 1, k] * pix_per_cpu)

                current_cpu = 0
                for k, vm_type in enumerate(type_list):
                    if current_pm_cpu[step, dest_pm, 0, k] != 0:
                        cv2.putText(x, f"{int(current_pm_cpu[step, dest_pm, 0, k]):02}",
                                    (4 * (pm_width + pm_gap) + 20 + numa_width // 8, last_y_start + current_cpu + 10),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                        current_cpu += int(current_pm_cpu[step, dest_pm, 0, k] * pix_per_cpu)

                current_cpu = 0
                for k, vm_type in enumerate(type_list):
                    if current_pm_cpu[step, dest_pm, 1, k] != 0:
                        cv2.putText(x, f"{int(current_pm_cpu[step, dest_pm, 1, k]):02}",
                                    (4 * (pm_width + pm_gap) + 20 + numa_width + numa_width // 8 + numa_gap,
                                     last_y_start + current_cpu + 10),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                        current_cpu += int(current_pm_cpu[step, dest_pm, 1, k] * pix_per_cpu)

                cv2.putText(x, f"new src",
                            (3 * (pm_width + pm_gap) + 20 + numa_width // 4, last_y_start + numa_height + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

                cv2.putText(x, f"new dst",
                            (4 * (pm_width + pm_gap) + 20 + numa_width // 4, last_y_start + numa_height + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

                if step != 0:
                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        if current_pm_cpu[step - 1, source_pm, 0, k] != 0:
                            cv2.putText(x, f"{int(current_pm_cpu[step - 1, source_pm, 0, k]):02}",
                                        (20 + numa_width // 8, last_y_start + current_cpu + 10),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                            current_cpu += int(current_pm_cpu[step - 1, source_pm, 0, k] * pix_per_cpu)

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        if current_pm_cpu[step - 1, source_pm, 1, k] != 0:
                            cv2.putText(x, f"{int(current_pm_cpu[step - 1, source_pm, 1, k]):02}",
                                        (20 + numa_width + numa_width // 8 + numa_gap,
                                         last_y_start + current_cpu + 10),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                            current_cpu += int(current_pm_cpu[step - 1, source_pm, 1, k] * pix_per_cpu)

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        if current_pm_cpu[step - 1, dest_pm, 0, k] != 0:
                            cv2.putText(x, f"{int(current_pm_cpu[step - 1, dest_pm, 0, k]):02}",
                                        (pm_width + pm_gap + 20 + numa_width // 8, last_y_start + current_cpu + 10),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                            current_cpu += int(current_pm_cpu[step - 1, dest_pm, 0, k] * pix_per_cpu)

                    current_cpu = 0
                    for k, vm_type in enumerate(type_list):
                        if current_pm_cpu[step - 1, dest_pm, 1, k] != 0:
                            cv2.putText(x, f"{int(current_pm_cpu[step - 1, dest_pm, 1, k]):02}",
                                        (pm_width + pm_gap + 20 + numa_width + numa_width // 8 + numa_gap,
                                         last_y_start + current_cpu + 10),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0))
                            current_cpu += int(current_pm_cpu[step - 1, dest_pm, 1, k] * pix_per_cpu)
                    cv2.putText(x, f"old src",
                                (20 + numa_width // 4, last_y_start + numa_height + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

                    cv2.putText(x, f"old dst",
                                (pm_width + pm_gap + 20 + numa_width // 4, last_y_start + numa_height + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

                cv2.putText(x, f"STEP {step}: {type_list[int(current_vm_types[step])]}",
                            ((pm_width + pm_gap) * 5 + 20 + numa_width // 4, last_y_start + numa_height // 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(x, f"NUMA{int(current_pm_involved[step, 2])} --> NUMA{int(current_pm_involved[step, 3])}",
                            ((pm_width + pm_gap) * 5 + 20 + numa_width // 4, last_y_start + numa_height // 4 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(x, f"Reward = {current_rewards[step]:.3f}, FR = {current_fr[step]:.3f}",
                            ((pm_width + pm_gap) * 5 + 20 + numa_width // 4, last_y_start + numa_height // 4 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(x, f"source PM involved steps: {involved_steps[current_pm_involved[step, 0]]}",
                            ((pm_width + pm_gap) * 5 + 20 + numa_width // 4, last_y_start + numa_height // 4 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(x, f"destination PM involved steps: {involved_steps[current_pm_involved[step, 1]]}",
                            ((pm_width + pm_gap) * 5 + 20 + numa_width // 4, last_y_start + numa_height // 4 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                frames.append(Image.fromarray(x, 'RGB'))
            frames[0].save(f"./runs/{run_name}/{restore_name}_env_{file_index + env_id}_trial_{trial_id}_"
                           f"FR_{current_fr[-1]:.3f}.gif", save_all=True,
                           append_images=frames[1:], optimize=False, duration=2000, loop=1)


if __name__ == "__main__":
    args = parse_args()
    num_train = args.num_train
    dev_idx_start = 6000
    num_dev = 200
    num_test = 200
    num_envs = args.num_envs
    num_steps = args.num_steps
    less_feature = args.less_feature
    run_name = f'{args.restore_name}'
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = AsyncVectorEnv_Patch(
        [make_env(args.gym_id, args.seed + i, args.vm_data_size, args.max_steps, num_train,
                  args.normalize, affinity=args.affinity) for i in range(num_envs)]
    )

    params = utils.Params(f'./experiments/pretrain/{args.model}/params.json')
    params.update(f'./data/params_{args.vm_data_size}.json')
    params.device = device
    params.batch_size = args.num_envs
    if less_feature:
        params.pm_cov = 6
        params.vm_cov = 10
    else:
        params.pm_cov = 8
        params.vm_cov = 14

    envs.call('set_save_json', save_json_flag=args.save_json, save_json_dir=os.path.join('runs', args.restore_name),
              save_json_file_name=args.restore_file)

    # input the vm candidate model
    vm_cand_model = models.VM_Extra_Sparse_Attn_Wrapper(params, args.pretrain).model
    pm_cand_model = models.PM_Detail_Attn_No_Flip_Wrapper(params).model

    agent = Agent(vm_cand_model, pm_cand_model, params, args.model, args.detail_pm)
    agent.eval()
    global_step = utils.load_checkpoint(args.restore_name, args.restore_file, agent)
    print(f"- Restored file (global step {global_step}) "
          f"from {os.path.join(args.restore_name, args.restore_file + '.pth.tar')}")

    if args.track:
        wandb.watch(agent, log_freq=100)

    obs_vm = torch.zeros(num_steps, args.num_envs, params.num_vm, params.vm_cov, device=device)
    obs_pm = torch.zeros(num_steps, args.num_envs, params.num_pm, params.pm_cov, device=device)
    obs_num_steps = torch.zeros(num_steps, args.num_envs, 1, 1, device=device)
    obs_num_vms = torch.zeros(num_steps, args.num_envs, dtype=torch.int32, device=device)
    vm_actions = torch.zeros(num_steps, args.num_envs, device=device)
    pm_actions = torch.zeros(num_steps, args.num_envs, device=device)
    logprobs = torch.zeros(num_steps, args.num_envs, device=device)
    rewards = torch.zeros(num_steps, args.num_envs, device=device)
    dones = torch.zeros(num_steps, args.num_envs, device=device)
    values = torch.zeros(num_steps, args.num_envs, device=device)
    # envs.single_action_space.nvec: [2089, 279] (#vm, #pm)
    action_masks = torch.zeros(num_steps, args.num_envs, envs.single_action_space.nvec[1], dtype=torch.bool,
                               device=device)

    pm_source_av = np.zeros((num_dev + num_test, num_steps))
    pm_source_fr = np.zeros((num_dev + num_test, num_steps))
    pm_dest_av = np.zeros((num_dev + num_test, num_steps))
    pm_dest_fr = np.zeros((num_dev + num_test, num_steps))
    all_pm_av = np.zeros((num_dev + num_test, num_steps, 5))
    all_pm_fr = np.zeros((num_dev + num_test, num_steps, 5))
    all_pm_av_nonzero = np.zeros((num_dev + num_test, num_steps))
    all_pm_fr_nonzero = np.zeros((num_dev + num_test, num_steps))
    all_rewards = np.zeros((num_dev + num_test, num_steps))

    with torch.no_grad():
        envs.call('set_mode', mode='dev')

        dev_all_frag_rate = np.ones((num_dev, num_steps))
        dev_all_min_frag_rate = np.ones((num_dev, num_steps))
        dev_pbar = trange(0, num_dev, num_envs, desc='Dev')
        for file_index in dev_pbar:
            file_ids = [dev_idx_start + file_index + env_id for env_id in range(num_envs)]
            envs.call_parse('set_current_env', env_id=file_ids)

            current_ep_info = np.zeros((num_steps, args.num_envs, 2)) + 1000  # return, len, fr
            pm_cpu_details = np.zeros((args.num_envs, num_steps, params.num_pm, 2, 8))
            pm_involved = np.zeros((args.num_envs, num_steps, 4))
            vm_type_step = np.zeros((args.num_envs, num_steps))
            next_obs_dict = envs.reset()
            next_obs_pm = torch.tensor(next_obs_dict['pm_info'], device=device)  # torch.Size([8, 279, 8])
            next_obs_vm = torch.tensor(next_obs_dict['vm_info'], device=device)  # torch.Size([8, 279, 14])
            if less_feature:
                next_obs_pm = torch.cat([next_obs_pm[:, :, :2],
                                         next_obs_pm[:, :, 4:]], dim=-1)
                next_obs_vm = torch.cat([next_obs_vm[:, :, :2], next_obs_vm[:, :, 4:8],
                                         next_obs_vm[:, :, 10:]], dim=-1)
            next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
            next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
            next_done = torch.zeros(args.num_envs, device=device)

            for step in range(0, num_steps):
                obs_pm[step] = next_obs_pm
                obs_vm[step] = next_obs_vm
                obs_num_steps[step] = next_obs_num_steps
                obs_num_vms[step] = next_obs_num_vms
                dones[step] = next_done

                vm_action, pm_action, logprob, _, value, action_mask \
                    = agent.get_action_and_value(envs, next_obs_pm, next_obs_vm, next_obs_num_steps, next_obs_num_vms)
                values[step] = value.flatten()  # value:  torch.Size([8, 1])
                action_masks[step] = action_mask
                vm_actions[step] = vm_action
                pm_actions[step] = pm_action
                logprobs[step] = logprob

                # print(f'vm_action: {vm_action.cpu().numpy()}, pm_action: {pm_action.cpu().numpy()}')
                next_obs_dict, reward, done, info = envs.step(torch.stack([vm_action, pm_action],
                                                                          dim=-1).cpu().numpy())
                next_obs_pm = torch.Tensor(next_obs_dict['pm_info']).to(device)
                next_obs_vm = torch.Tensor(next_obs_dict['vm_info']).to(device)
                if less_feature:
                    next_obs_pm = torch.cat([next_obs_pm[:, :, :2],
                                             next_obs_pm[:, :, 4:]], dim=-1)
                    next_obs_vm = torch.cat([next_obs_vm[:, :, :2], next_obs_vm[:, :, 4:8],
                                             next_obs_vm[:, :, 10:]], dim=-1)
                next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
                next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                all_rewards[file_index:file_index + num_envs, step] = reward
                next_done = torch.Tensor(done).to(device)

                for env_id, item in enumerate(info):
                    dev_all_frag_rate[file_index + env_id, step] = item['fragment_rate']
                    pm_source_av[file_index + env_id, step] = item['pm_source_av']
                    pm_source_fr[file_index + env_id, step] = item['pm_source_fr']
                    pm_dest_av[file_index + env_id, step] = item['pm_dest_av']
                    pm_dest_fr[file_index + env_id, step] = item['pm_dest_fr']
                    all_pm_fr[file_index + env_id, step] = item['all_pm_fr']
                    all_pm_av[file_index + env_id, step] = item['all_pm_av']
                    all_pm_av_nonzero[file_index + env_id, step] = item['all_pm_av_nonzero']
                    all_pm_fr_nonzero[file_index + env_id, step] = item['all_pm_fr_nonzero']
                    current_ep_info[step, env_id, 1] = item['fragment_rate']
                    pm_cpu_details[env_id, step] = item['pm_cpu_details']
                    pm_involved[env_id, step] = item['pm_involved']
                    vm_type_step[env_id, step] = item['vm_type']
                    if "episode" in item.keys():
                        current_ep_info[step, env_id, 0] = item["episode"]["r"]
                        dev_all_min_frag_rate[file_index + env_id, step] = item['fragment_rate']

            plot_pm_details(dev_all_frag_rate[file_index:file_index + num_envs],
                            all_rewards[file_index:file_index + num_envs],
                            pm_cpu_details, pm_involved, vm_type_step, args.max_steps, params.num_pm, run_name,
                            args.restore_file, file_index)

        for i in range(num_dev):
            print(f'dev {i}: {dev_all_min_frag_rate[i][dev_all_min_frag_rate[i] != 1]}')
        current_dev_frag_rate = np.mean(np.amin(dev_all_min_frag_rate, axis=1))
        np.save(os.path.join('runs', args.restore_name, 'dev_all_frag_rate.npy'), dev_all_frag_rate)
        print(f'Dev fragment rate: {current_dev_frag_rate:.4f}')

        envs.call('set_mode', mode='test')

        test_all_min_frag_rate = np.ones((num_test, num_steps))
        test_pbar = trange(0, num_test, num_envs, desc='Test')
        for file_index in test_pbar:
            file_ids = [dev_idx_start + num_dev + file_index + env_id for env_id in range(num_envs)]
            envs.call_parse('set_current_env', env_id=file_ids)

            current_ep_info = np.zeros((num_steps, args.num_envs, 2)) - 1000  # return, len, fr
            next_obs_dict = envs.reset()
            next_obs_pm = torch.tensor(next_obs_dict['pm_info'], device=device)  # torch.Size([8, 279, 8])
            next_obs_vm = torch.tensor(next_obs_dict['vm_info'], device=device)  # torch.Size([8, 279, 14])
            if less_feature:
                next_obs_pm = torch.cat([next_obs_pm[:, :, :2],
                                         next_obs_pm[:, :, 4:]], dim=-1)
                next_obs_vm = torch.cat([next_obs_vm[:, :, :2], next_obs_vm[:, :, 4:8],
                                         next_obs_vm[:, :, 10:]], dim=-1)
            next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
            next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
            next_done = torch.zeros(args.num_envs, device=device)

            for step in range(0, num_steps):
                obs_pm[step] = next_obs_pm
                obs_vm[step] = next_obs_vm
                obs_num_steps[step] = next_obs_num_steps
                obs_num_vms[step] = next_obs_num_vms
                dones[step] = next_done

                vm_action, pm_action, logprob, _, value, action_mask \
                    = agent.get_action_and_value(envs, next_obs_pm, next_obs_vm, next_obs_num_steps, next_obs_num_vms)
                values[step] = value.flatten()  # value:  torch.Size([8, 1])
                action_masks[step] = action_mask
                vm_actions[step] = vm_action
                pm_actions[step] = pm_action
                logprobs[step] = logprob

                # print(f'vm_action: {vm_action.cpu().numpy()}, pm_action: {pm_action.cpu().numpy()}')
                next_obs_dict, reward, done, info = envs.step(torch.stack([vm_action, pm_action],
                                                                          dim=-1).cpu().numpy())
                next_obs_pm = torch.Tensor(next_obs_dict['pm_info']).to(device)
                next_obs_vm = torch.Tensor(next_obs_dict['vm_info']).to(device)
                if less_feature:
                    next_obs_pm = torch.cat([next_obs_pm[:, :, :2],
                                             next_obs_pm[:, :, 4:]], dim=-1)
                    next_obs_vm = torch.cat([next_obs_vm[:, :, :2], next_obs_vm[:, :, 4:8],
                                             next_obs_vm[:, :, 10:]], dim=-1)
                next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
                next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                all_rewards[num_dev + file_index:num_dev + file_index + num_envs, step] = reward
                next_done = torch.Tensor(done).to(device)

                for env_id, item in enumerate(info):
                    pm_source_av[num_dev + file_index + env_id, step] = item['pm_source_av']
                    pm_source_fr[num_dev + file_index + env_id, step] = item['pm_source_fr']
                    pm_dest_av[num_dev + file_index + env_id, step] = item['pm_dest_av']
                    pm_dest_fr[num_dev + file_index + env_id, step] = item['pm_dest_fr']
                    all_pm_fr[num_dev + file_index + env_id, step] = item['all_pm_fr']
                    all_pm_av[num_dev + file_index + env_id, step] = item['all_pm_av']
                    all_pm_av_nonzero[num_dev + file_index + env_id, step] = item['all_pm_av_nonzero']
                    all_pm_fr_nonzero[num_dev + file_index + env_id, step] = item['all_pm_fr_nonzero']
                    current_ep_info[step, env_id, 1] = item['fragment_rate']
                    if "episode" in item.keys():
                        current_ep_info[step, env_id, 0] = item["episode"]["r"]
                        test_all_min_frag_rate[file_index + env_id, step] = item['fragment_rate']

        current_test_frag_rate = np.mean(np.amin(test_all_min_frag_rate, axis=1))
        print(f'Test fragment rate: {current_test_frag_rate:.4f}')

        np.save(f"runs/{run_name}/{args.restore_file}_dev_all_min_frag_rate.npy", dev_all_min_frag_rate)
        np.save(f"runs/{run_name}/{args.restore_file}_test_all_min_frag_rate.npy", test_all_min_frag_rate)
        np.save(f"runs/{run_name}/{args.restore_file}_pm_source_av.npy", pm_source_av)
        np.save(f"runs/{run_name}/{args.restore_file}_pm_source_fr.npy", pm_source_fr)
        np.save(f"runs/{run_name}/{args.restore_file}_pm_dest_av.npy", pm_dest_av)
        np.save(f"runs/{run_name}/{args.restore_file}_pm_dest_fr.npy", pm_dest_fr)
        np.save(f"runs/{run_name}/{args.restore_file}_all_pm_fr.npy", all_pm_fr)
        np.save(f"runs/{run_name}/{args.restore_file}_all_pm_av.npy", all_pm_av)
        np.save(f"runs/{run_name}/{args.restore_file}_all_pm_fr_nonzero.npy", all_pm_fr_nonzero)
        np.save(f"runs/{run_name}/{args.restore_file}_all_pm_av_nonzero.npy", all_pm_av_nonzero)
        np.save(f"runs/{run_name}/{args.restore_file}_all_rewards.npy", all_rewards)

    envs.close()
