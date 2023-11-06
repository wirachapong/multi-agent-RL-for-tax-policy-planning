# main.py
from environment_1nn import Environment_1nn
from policyplanneragent import PolicyPlannerAgent
from person_1nn import Person_1nn
from constants import NUM_PERSONS
from double_auction import *

def main():
    
    
    EPSILON = 0.1  # Consider moving constants to a separate config file or module
    total_reward_policy_planner = 0
    total_reward_individual = 0
    NUM_EPISODES = 1000  # You might need more episodes for training
    env = Environment_1nn(NUM_PERSONS)


    for episode in range(NUM_EPISODES):
        if episode == NUM_EPISODES - 1:
            is_terminal_state = True
        else:
            is_terminal_state = False
        
        print('Episode', episode)
        
        reward_policy_planner, reward_individual = simulate_episode(env, is_terminal_state)
        total_reward_policy_planner += reward_policy_planner
        total_reward_individual += reward_individual
        # Optionally decrease epsilon over time to reduce exploration
        if EPSILON > 0.01:
            EPSILON *= 0.995  

    print(f"Total reward after {NUM_EPISODES} episodes: {[total_reward_policy_planner,total_reward_individual]}")

# Hino: I currently set the action flow of the double auction to be done in the simulate episode too
#  but it will be done as the very first step of each episode so that the result of those auctions will also be included in the reward of each episode. 

def simulate_episode(env, is_terminal_state=False):
    current_state = env.get_state()
    
    #! Either this in main.py or in Environment.py
    # #  Here should be the space that each individual start doing actions
    # for person in env.persons:
    #     current_state = person.get_state()
    #     person_action = person.select_action()

    #     person.take_action(person_action, env.policy_planner_agent.tax_rate_for_income)
        
    #     person_reward = person.get_reward()
    #     person_next_state = person.get_state()

    #     person.remember(current_state, person_action, person_reward, person_next_state)
        
    #     #! Maybe not do this to batch training later instead
    #     person.replay()
    next_state0= env.persons_gain_category_token()
    next_state1= env.fill_random_action_history()
    next_state2= env.persons_do_bid_sell() # learn of buying and selling is already included in here
    action = env.PolicyPlannerAgent.select_action(next_state2)
    print(action)
    total_cost = env.PolicyPlannerAgent.apply_action(action, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
    next_state2 = env.persons_step(is_terminal_state) # all persons learn or earn and tax is collected. 
    reward_policy_planner = env.PolicyPlannerAgent.get_reward(0, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
    
    # we used 0 for now in the (a,b) for previously used get_reward function due to how there's a change in how the policy changed from our first structure
    env.PolicyPlannerAgent.remember(current_state, action, reward_policy_planner, next_state2)
    env.PolicyPlannerAgent.replay()  # Experience replay

    total_reward_individual = sum([person.get_reward() for person in env.persons])
    return [reward_policy_planner, total_reward_individual]

if __name__ == "__main__":
    main()
