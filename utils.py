import random
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import configuration
import matplotlib.patches as mpatches



def get_discount_rate_heuristic(name: str):
    if name == "random_dist_0_10":
        return lambda : random.randrange(20) / 100.0
    if name =="none":
        return lambda : 0


def plot_education_for_cycle(education_data: List[List[int]]):
    # plt.figure(figsize=(10, 6))
    # for episode in range(len(education_data)):
    #     education_levels = education_data[episode]
    #     colors = [plt.cm.viridis(level / 7) for level in
    #               education_levels]  # Color-coded based on education level
    #     plt.bar(range(len(education_data[0])), education_levels, color=colors, alpha=0.6)
    #     plt.title(f"Education Levels in Episode {episode + 1}")
    #     plt.xlabel("Agent")
    #     plt.ylabel("Education Level (1-7)")

    # Customize the graph, add labels, legends, etc.
    # plt.figure(figsize=(10, 6))
    #
    # unique_education_levels = np.array(configuration.config.get_constant("EDUCATION_LEVELS"))
    # colors = [plt.cm.viridis(level / 7) for level in unique_education_levels]
    # legend_patches = []
    #
    # for episode in range(len(education_data)):
    #     education_levels = education_data[episode]
    #
    #     segments = []
    #     for level in unique_education_levels:
    #         count = np.sum(education_levels == level)
    #         segments.append(count)
    #
    #     plt.bar(episode, segments, bottom=np.cumsum(segments) - segments[0],
    #             color=colors, edgecolor='none')
    #
    # for level, color in zip(unique_education_levels, colors):
    #     legend_patches.append(
    #         mpatches.Patch(color=color, label=f"Education Level {level}"))
    #
    # plt.legend(handles=legend_patches, loc='upper right')
    #
    # plt.title("Education Levels Over Episodes")
    # plt.xlabel("Episode")
    # plt.ylabel("Number of Agents")
    #
    # plt.grid()
    # return plt
    #
    #
    # Customize the graph, add labels, legends, etc.
    plt.figure(figsize=(10, 6))

    episodes = len(education_data)

    # Assuming unique education levels are integers from 0 to max level found in the data
    all_levels = [level for episode in education_data for level in episode]
    unique_education_levels = list(range(max(all_levels) + 1))

    colors = [plt.cm.viridis(level / len(unique_education_levels)) for level in
              unique_education_levels]
    legend_patches = []

    for episode_idx, levels in enumerate(education_data):
        counts = [levels.count(level) for level in unique_education_levels]

        plt.bar(episode_idx, counts, bottom=np.cumsum(counts) - counts,
                color=colors, edgecolor='none')

    for level, color in zip(unique_education_levels, colors):
        legend_patches.append(
            mpatches.Patch(color=color, label=f"Education Level {level}"))

    plt.legend(handles=legend_patches, loc='upper right')

    plt.title("Education Levels Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Number of Agents")

    # Set y-axis to length equal to num_agents
    # plt.ylim(0, num_agents)

    plt.grid(True)
    return plt

def plot_reward(rewards, window_size):
    # Calculate the moving average over the past 5 iterations
    moving_average = [np.mean(rewards[i:i + window_size]) for i in
                      range(len(rewards) - window_size + 1)]
    x = list(range(1, len(rewards) + 1))

    # Create a line plot for the original data
    plt.plot(x, rewards, label='Data')

    # Create a line plot for the moving average
    x_ma = list(range(window_size, len(rewards) + 1))
    plt.plot(x_ma, moving_average, label='Moving Average (Window Size = 5)')

    # Add labels to the axes
    plt.xlabel('Iteration')
    plt.ylabel('Value')

    # Add a title to the plot
    plt.title('Data and Moving Average')

    # Add a legend
    plt.legend()
    return plt

