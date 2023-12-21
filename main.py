# main.py
import argparse
import os
import shutil
from datetime import datetime

import configuration
import utils
from configuration import Configuration
from environment import Environment
from environment_0nn import Environment_0nn
from environment_1nn import Environment_1nn


def main():
    parser = argparse.ArgumentParser(description="Your Application Description")
    # Add the "--config-file" argument with a default value
    parser.add_argument(
        "--config-file",
        help="Path to the configuration file",
        default="config.json"  # Specify the default file location here
    )

    args = parser.parse_args()

    configuration.config = Configuration(args.config_file)
    NUM_LIFECYCLES = configuration.config.get_constant("NUM_LIFECYCLES")
    NUM_PERSONS = configuration.config.get_constant("NUM_PERSONS")
    NUM_EPISODES = configuration.config.get_constant("NUM_EPISODES")

    if configuration.config.get_constant("0nn"):
        env = Environment_0nn(NUM_PERSONS, NUM_EPISODES)  # With best response agents

    elif configuration.config.get_constant("1nn"):
        env = Environment_1nn(NUM_PERSONS)  # With 1 neural network for persons

    else:
        env = Environment(NUM_PERSONS)  # With neural network for each person

    rewards = []

    # Can be set in the config.json. runs the model with random start points in order
    # to learn before running the main task.
    for learning_cycle in range(configuration.config.get_constant("LEARNING_CYCLES")):
        env.set_policy_planer_tax(utils.create_random_tax_brackets(
            len(configuration.config.get_constant("START_TAX_RATE"))))
        for lifecycles_in_lc in range(
                configuration.config.get_constant("LIFE_CYCLES_IN_LC")):
            if lifecycles_in_lc % 5 == 0:
                print(f"LEARNING CYCLE: {learning_cycle}, LIFECYCLE: {lifecycles_in_lc}")
            env.simulate_lifecycle(NUM_EPISODES, True, lifecycles_in_lc != 0, False)

    # Set the start tax policy
    env.set_policy_planer_tax(configuration.config.get_constant("START_TAX_RATE"))

    # Run the model starting from start run policy
    for lifecycle in range(NUM_LIFECYCLES):
        if lifecycle % 5 == 0:
            print(f"LIFECYCLE: {lifecycle}")
        env.simulate_lifecycle(NUM_EPISODES,
                               lifecycle < configuration.config.get_constant(
                                   "EPSILON_ROUNDS_POLICY"), lifecycle != 0)

    save_model(NUM_LIFECYCLES, env)


def save_model(NUM_LIFECYCLES, env):
    if not os.path.exists(configuration.config.get_constant('OUTPUT_FOLDER_PATH')):
        os.mkdir(configuration.config.get_constant('OUTPUT_FOLDER_PATH'))

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    directory_name = f"{formatted_datetime}"
    save_model_path = f"{configuration.config.get_constant('OUTPUT_FOLDER_PATH')}/{directory_name}"
    os.mkdir(save_model_path)

    utils.plot_reward(env.saved_data["rewards_per_cycle"], 5).savefig(
        f"{save_model_path}/rewards_per_cycle")
    for i in range(0, NUM_LIFECYCLES, int(NUM_LIFECYCLES / 10)):
        plot = utils.plot_education_for_cycle(env.saved_data["person_educations"][i])
        plot.savefig(f"{save_model_path}/education_level_cycle_{i}.png")
    plot = utils.plot_education_for_cycle(
        env.saved_data["person_educations"][NUM_LIFECYCLES - 1])
    plot.savefig(
        f"{save_model_path}/education_level_cycle_{NUM_LIFECYCLES - 1}_FINAL.png")
    env.save_policy_planner(f"{save_model_path}/lifecycle_{NUM_LIFECYCLES}_")
    shutil.copy("config.json", f"{save_model_path}/config.json")


if __name__ == "__main__":
    main()
