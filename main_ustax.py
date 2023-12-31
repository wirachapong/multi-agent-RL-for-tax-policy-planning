# main.py
import shutil

import configuration
import utils
from environment import Environment
from environment_0nn import Environment_0nn
from environment_1nn import Environment_1nn
import argparse
from configuration import Configuration, config
import os
from datetime import datetime



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
        env = Environment_0nn(NUM_PERSONS, NUM_EPISODES)    # With best response agents
    
    elif configuration.config.get_constant("1nn"):
        env = Environment_1nn(NUM_PERSONS)                # With 1 neural network for persons
    
    else:
        env = Environment(NUM_PERSONS)                     # With neural network for each person

    rewards = []
    for lifecycle in range(NUM_LIFECYCLES):
        if lifecycle%5 == 0:
            print(f"LIFECYCLE: {lifecycle}")
        env.simulate_lifecycle(NUM_EPISODES, lifecycle< configuration.config.get_constant("EPSILON_ROUNDS_POLICY"))

    save_model(NUM_LIFECYCLES, env)


def save_model(NUM_LIFECYCLES, env):
    if not os.path.exists(configuration.config.get_constant('OUTPUT_FOLDER_PATH')):
        os.mkdir(configuration.config.get_constant('OUTPUT_FOLDER_PATH'))

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    directory_name = f"{formatted_datetime}"
    save_model_path = f"{configuration.config.get_constant('OUTPUT_FOLDER_PATH')}/{directory_name}"
    os.mkdir(save_model_path)

    utils.plot_reward(env.saved_data["rewards_per_cycle"], 5).savefig(f"{save_model_path}/rewards_per_cycle")
    for i in range(0, NUM_LIFECYCLES, int(NUM_LIFECYCLES/10)):
        plot = utils.plot_education_for_cycle(env.saved_data["person_educations"][i])
        plot.savefig(f"{save_model_path}/education_level_cycle_{i}.png")
    plot = utils.plot_education_for_cycle(
        env.saved_data["person_educations"][NUM_LIFECYCLES - 1])
    plot.savefig(f"{save_model_path}/education_level_cycle_{NUM_LIFECYCLES - 1}_FINAL.png")
    env.save_policy_planner(f"{save_model_path}/lifecycle_{NUM_LIFECYCLES}_")
    shutil.copy("config.json", f"{save_model_path}/config.json")


if __name__ == "__main__":
    main()

    
#! OLD CODE 

    # for episode in range(NUM_EPISODES):
    #     if episode == NUM_EPISODES - 1:
    #         is_terminal_state = True
    #     else:
    #         is_terminal_state = False

    #     print('Episode', episode)

    #     reward_policy_planner, reward_individual = env.simulate_episode(is_terminal_state)
    #     total_reward_policy_planner += reward_policy_planner
    #     total_reward_individual += reward_individual
    #     # Optionally decrease epsilon over time to reduce exploration
    #     if EPSILON > 0.01:
    #         EPSILON *= 0.995

    # print(f"Total reward after {NUM_EPISODES} episodes: {[total_reward_policy_planner,total_reward_individual]}")

# Hino: I currently set the action flow of the double auction to be done in the simulate episode too
#  but it will be done as the very first step of each episode so that the result of those auctions will also be included in the reward of each episode.

# def simulate_episode(env, is_terminal_state=False):

#     current_state = env.get_state()
#     #! Either this in main.py or in Environment.py
#     # #  Here should be the space that each individual start doing actions
#     # for person in env.persons:
#     #     current_state = person.get_state()
#     #     person_action = person.select_action()

#     #     person.take_action(person_action, env.policy_planner_agent.tax_rate_for_income)
        
#     #     person_reward = person.get_reward()
#     #     person_next_state = person.get_state()

#     #     person.remember(current_state, person_action, person_reward, person_next_state)
        
#     #     #! Maybe not do this to batch training later instead
#     #     person.replay()
#     next_state0= env.persons_gain_category_token()

#     next_state1= env.fill_random_action_history()
    

#     next_state2= env.persons_do_bid_sell() # learn of buying and selling is already included in here

    # current_state = env.get_state()

#     next_state3= env.bid_sell_system.clear_previous_round()

#     action = env.PolicyPlannerAgent.select_action(next_state3)

#     print(["%.2f" % tax_rate for tax_rate in env.PolicyPlannerAgent.current_tax_rate])
#     total_cost = env.PolicyPlannerAgent.apply_action(action, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
#     next_state2 = env.persons_step(is_terminal_state) # all persons learn or earn and tax is collected.
#     reward_policy_planner = env.PolicyPlannerAgent.get_reward(0, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
    
#     # we used 0 for now in the (a,b) for previously used get_reward function due to how there's a change in how the policy changed from our first structure
#     env.PolicyPlannerAgent.remember(current_state, action, reward_policy_planner, next_state2)
#     env.PolicyPlannerAgent.replay()  # Experience replay
#     env.bid_sell_system.end_round()
#     total_reward_individual = sum([person.get_reward() for person in env.persons])
#     return [reward_policy_planner, total_reward_individual]
#     action = env.PolicyPlannerAgent.select_action(current_state)
#     total_cost = env.PolicyPlannerAgent.apply_action(action, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
#     next_state2 = env.persons_step(is_terminal_state) # all persons learn or earn and tax is collected.
#     reward_policy_planner = env.PolicyPlannerAgent.get_reward(0, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
#     # we used 0 for now in the (a,b) for previously used get_reward function due to how there's a change in how the policy changed from our first structure
#     env.PolicyPlannerAgent.remember(current_state, action, reward_policy_planner, next_state2)
#     env.PolicyPlannerAgent.replay()  # Experience replay
#     env.remove_redundant_current_dict()
#     total_reward_individual = sum([person.get_reward() for person in env.persons])
#     return [reward_policy_planner, total_reward_individual]
# >>>>>>> Stashed changes

