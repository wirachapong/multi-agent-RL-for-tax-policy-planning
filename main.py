# main.py
from environment import Environment
from policyplanneragent import PolicyPlannerAgent

def main():
    env = Environment()
    EPSILON = 0.1  # Consider moving constants to a separate config file or module
    total_reward = 0
    num_episodes = 5000  # You might need more episodes for training

    for episode in range(num_episodes):
        print('Episode', episode)
        total_reward += simulate_episode(env)

        # Optionally decrease epsilon over time to reduce exploration
        if EPSILON > 0.01:
            EPSILON *= 0.995

    print(f"Total reward after {num_episodes} episodes: {total_reward}")

def simulate_episode(env):
    current_state = env.get_state()
    action = env.PolicyPlannerAgent.select_action(current_state)
    total_cost = env.PolicyPlannerAgent.apply_action(action, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
    next_state = env.step(action)  # Adjusted for simplicity; step might need more info
    reward = env.PolicyPlannerAgent.get_reward(0, env.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
    # we used 0 for now in the (a,b) for previously used get_reward function due to how there's a change in how the policy changed from our first structure
    env.PolicyPlannerAgent.remember(current_state, action, reward, next_state)
    env.PolicyPlannerAgent.replay()  # Experience replay
    return reward

if __name__ == "__main__":
    main()
