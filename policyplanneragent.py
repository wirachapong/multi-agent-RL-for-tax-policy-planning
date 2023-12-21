from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import configuration
from person import Person

TAX_CHANGE_DICT = {4: -1, 3: -.1, 0: 0 , 1: .1, 2: 1}
class QNetwork(nn.Module):
    """
    Q-netwrok for PolicyPlannerAgent
    """
    def __init__(self, input_dim: int, num_actions: int):
        super(QNetwork, self).__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7 * num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.num_actions, 7)


class PolicyPlannerAgent:
    """
    makes changes to tax brackets in order to optimize his reward.
    """
    def __init__(self, input_dim: int, num_actions: int):
        self.model = QNetwork(input_dim, num_actions)
        self.current_tax_rate = configuration.config.get_constant("START_TAX_RATE")
        self.memory = deque()  # For experience replay
        self.history_of_auctions = []
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=configuration.config.get_constant("ALPHA_POLICY"))
        self.EPSILON = configuration.config.get_constant("EPSILON_POLICY")
        self.num_moves = 0
        self.count = 0

    def select_action(self, state):
        valid_actions = []
        if (np.random.uniform(0, 1) < self.EPSILON
                or len(self.memory) < configuration.config.get_constant(
                    "MEMORY_SIZE_POLICY")):
            # Ensure that the random action will not result in tax rates outside the 0-80% range.
            for tax_rate in self.current_tax_rate:
                available_actions = [action for action, change in TAX_CHANGE_DICT.items()
                                     if 0 <= change + tax_rate <= 80]
                assert (0 in available_actions)
                valid_actions.append(np.random.choice(available_actions))
            return torch.tensor(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor)
                max_values, max_indices = torch.max(q_values, dim=1)
                # Enforce valid actions (Prevent tax rate from going beyond 0% or above 80%)
                valid_actions = []
                for i, tax_rate in enumerate(self.current_tax_rate):
                    if 0 <= tax_rate + TAX_CHANGE_DICT[max_indices[0][i].item()] <= 80:
                        valid_actions.append(max_indices[0][i].item())
                    else:
                        valid_actions.append(
                            0)  # Replace with '0' to hold tax rate constant

        return torch.tensor(valid_actions)


    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > configuration.config.get_constant("MEMORY_SIZE_POLICY"):
            self.memory.popleft()

    def replay(self):
        if len(self.memory) < configuration.config.get_constant("BATCH_SIZE_POLICY"):
            return

        entropy_coef = configuration.config.get_constant(
            "ENTROPY_COEF")

        batch_indices = np.random.choice(len(self.memory),
                                         configuration.config.get_constant(
                                             "BATCH_SIZE_POLICY"), replace=False)
        batch = [self.memory[i] for i in batch_indices]

        for state, action, reward, next_state in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            with torch.no_grad():
                future_q_values = self.model(next_state_tensor)
                max_future_q_values, _ = torch.max(future_q_values, dim=1)
                target = reward + configuration.config.get_constant(
                    "GAMMA_POLICY") * max_future_q_values

            current_q_values = self.model(state_tensor)
            current_q_value = current_q_values[0][action]

            loss = nn.MSELoss()(current_q_value, target)

            # Compute the entropy of the policy (negative log of softmax probabilities of actions)
            log_softmax_actions = F.log_softmax(current_q_values, dim=1)
            softmax_actions = torch.exp(log_softmax_actions)
            entropy = -(softmax_actions * log_softmax_actions).sum()

            loss -= entropy_coef * entropy

            # Perform a gradient descent step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def apply_action(self, action, persons):
        self.num_moves += 1
        total_cost = 0
        current_tax_rate = self.current_tax_rate
        new_tax_rate = [
            rate + TAX_CHANGE_DICT[act.item()]
            for rate, act in zip(current_tax_rate, action)]
        self.current_tax_rate = new_tax_rate
        return total_cost

    def tax_rate_for_income(self, income):
        """

        Args:
            income: income of a Person

        Returns: breaks the income into net-income, tax

        """

        brackets = self.current_tax_rate
        BRACKET_GAP = configuration.config.get_constant("BRACKET_GAP")
        income_bracket_index = min(int(income / BRACKET_GAP), len(brackets) - 1)


        # To only tax the remaining income in the last bracket
        income_over_last_index = income - income_bracket_index * BRACKET_GAP
        tax_over_last_index = income_over_last_index * (brackets[income_bracket_index]/100)
        tax_income = tax_over_last_index
        
        for i in range(income_bracket_index):
            tax_rate = max(min(brackets[i] / 100, 1), 0)
            tax_income += tax_rate * BRACKET_GAP

        net_income = income - tax_income

        return net_income, tax_income

    def get_gini(self, persons):
        """
        Calculates gini. Currently not used.
        """

        x = np.array([person.net_worth for person in persons])
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x) ** 2 * np.mean(x))

    def get_reward(self, total_cost: float, persons: List[Person], is_terminal_state=True)-> float:
        """
        Calculates the PolicyPlanner reward.

        Args:
            total_cost: cost of PolicyPlanner operation
            persons: list of Persons in the system
            is_terminal_state: True if last state in a round

        Returns: reward

        """
        if is_terminal_state:
            gini = self.get_gini(persons) # not used

            net_worth_sum = sum([person.net_worth for person in persons])
            reward = net_worth_sum - total_cost
        else:
            reward = 0
        return reward

    def reset(self):
        """
        resets the system after each round
        """

        pass
