# multi-agent-RL-for-tax-policy-planning README

## Overview

This simulation application models the life cycles of individuals in a dynamic
environment, considering education, earnings, taxes, and various economic factors.
Reinforcement learning techniques are used to simulate decision-making processes and
policy planning.

## Features

- **Education System:**
    - Different education levels with associated turns required and earnings.

- **Agents:**
    - Supports two types of environments, in each one PolicyPlanner is a learning agent:
        - `0nn`: Best response Person agents.
        - `1nn`: One neural network for each Person.


- **Environment Parameters:**
    - `EDUCATION_LEVELS`: Possible levels of education for individuals.
    - `EDUCATION_TURNS_REQUIRED`: Turns required to increase education level.
    - `EDUCATION_EARNINGS`: Earning for each education level.
    - `COST_OF_LIVING`: Cost of living for individuals.
    - `NETWORTH_TURN0`: Initial net worth at the start.
    - `NUM_PERSONS`: Number of individuals in the simulation.
    - `NUM_EPISODES`: Number of episodes in each lifecycle.
    - `NUM_LIFECYCLES`: Total number of lifecycles in the simulation.
    - `LEARNING_CYCLES`: Number of learning cycles.
    - `LIFE_CYCLES_IN_LC`: Number of lifecycles in each learning cycle.

- **Policy Planning:**
    - Policy planning parameters for reinforcement learning:
        - `ALPHA_POLICY`, `GAMMA_POLICY`, `EPSILON_POLICY`, etc.

- **Person Reinforcement Learning:**
    - Reinforcement learning parameters for individuals:
        - `ALPHA_PERSON`, `GAMMA_PERSON`, `EPSILON_PERSON`, etc.

- **Economic Factors:**
    - Economic factors such as base salary, education increase, and discount rate
      heuristic.

- **Taxation:**
    - Tax-related parameters and starting tax rates.

- **Output:**
    - Plots and data saved in the specified output folder.
    - Reward plots, education level plots, and policy planner saved for analysis.

## Configuration

The application utilizes a configuration file (`config.json`) for easy customization.
Users can adjust parameters such as education levels, environment type, reinforcement
learning settings, and more. A sample named `config_sample.json` is included in the git
repository.

## Usage

1. Install dependencies: Ensure that the required Python libraries are
   installed (`matplotlib`, `numpy`, etc.).

2. Configure settings: Edit the `config_sample.json` file to customize simulation
   parameters.

3. Run the simulation: Execute the `main.py` script to start the simulation.
    ```bash 
       python main.py --config-file config.json
4. Analyze results: Examine the output folder for saved plots and data.

## Output

The simulation generates informative visualizations, including reward plots and education
level plots for each lifecycle. Additionally, policy planners and the configuration file
are saved for further analysis.




