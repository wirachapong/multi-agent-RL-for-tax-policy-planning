# main.py
from environment_1nn import Environment_1nn
from policyplanneragent import PolicyPlannerAgent
from person_1nn import Person_1nn
from constants import NUM_PERSONS

def main():
    
    env = Environment_1nn(NUM_PERSONS)
    EPSILON = 0.1  # Consider moving constants to a separate config file or module
    total_reward_policy_planner = 0
    total_reward_individual = 0
    num_episodes = 100  # You might need more episodes for training

    for episode in range(num_episodes):
        print('Episode', episode)
        reward_policy_planner, reward_individual = simulate_episode(env)
        total_reward_policy_planner += reward_policy_planner
        total_reward_individual += reward_individual
        # Optionally decrease epsilon over time to reduce exploration
        if EPSILON > 0.01:
            EPSILON *= 0.995  

    print(f"Total reward after {num_episodes} episodes: {[total_reward_policy_planner,total_reward_individual]}")

def simulate_episode(env):
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

    action = env.PolicyPlannerAgent.select_action(current_state)
    print(action)
    total_cost = env.PolicyPlannerAgent.apply_action(action, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
    next_state = env.persons_step()
    reward_policy_planner = env.PolicyPlannerAgent.get_reward(0, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
    
    # we used 0 for now in the (a,b) for previously used get_reward function due to how there's a change in how the policy changed from our first structure
    env.PolicyPlannerAgent.remember(current_state, action, reward_policy_planner, next_state)
    env.PolicyPlannerAgent.replay()  # Experience replay

    total_reward_individual = sum([person.get_reward() for person in env.persons])
    return [reward_policy_planner, total_reward_individual]

if __name__ == "__main__":
    main()
