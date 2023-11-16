
import configuration
import utils
from environment import Environment
from person_0nn import Person_0nn
from policyplanneragent import PolicyPlannerAgent
import random
from double_auction import *
from bid_sell import *
from environment_0nn import Environment_0nn


class us_tax_rates(Environment_0nn):
    def __init__(self, n_persons:int, horizon: int, random_seed = 1):
        super().__init__(n_persons, random_seed = random_seed)
        self.available_category_of_person = ["A","B","C"]
        
        # For person decisions
        self.horizon = horizon
        self.time_step = 0
        education_level_turn0 = configuration.config.get_constant("EDUCATION_LEVELS")
        net_worth_turn0 = configuration.config.get_constant("NETWORTH_TURN0")
        n_brackets = configuration.config.get_constant("N_BRACKETS")
        self.discount_rate_func = utils.get_discount_rate_heuristic(configuration.config.get_constant("DISCOUNT_RATE_HEURISTIC"))

        random.seed(self.random_seed)
        self.persons = [Person_0nn(idx, random.choice(self.education_level_turn0), self.net_worth_turn0, category=random.choice(self.available_category_of_person), discount_rate=self.discount_rate_func()) for idx in range(n_persons)]

        #self.PolicyPlannerAgent = PolicyPlannerAgent(2 * n_persons + n_brackets, len(configuration.config.get_constant("ACTIONS")))
        self.PolicyPlannerAgent = None

        self.bid_sell_system = BidSellSystem(commodities=self.available_category_of_person ,agents=self.persons)


    def us_tax_brackets(self):
        return [10,12,22,24,32,35,37]

    def tax_rate_for_income(self, income):

        brackets = self.us_tax_brackets()
        BRACKET_GAP = configuration.config.get_constant("BRACKET_GAP")
        income_bracket_index = min(int(income / BRACKET_GAP), len(brackets))

        # income_over_last_index = income - income_bracket_index * BRACKET_GAP
        # tax_over_last_index = income_over_last_index * (brackets[income_bracket_index]/100)

        # tax_income = tax_over_last_index
        tax_income = 0
        for i in range(income_bracket_index):
            tax_rate = max(min(brackets[i] / 100, 1), 0)
            tax_income += tax_rate * BRACKET_GAP

        person_income = income - tax_income

        return person_income, tax_income

    def persons_step(self, is_terminal_state=False):

        # for concurrent execution- in my PC is longer than reg run
        # person_actions = self.get_person_actions_concurrently()

        person_actions = [person.select_action(self.time_step, self.horizon, self.tax_rate_for_income) for person in self.persons]

        for action, person in zip(person_actions, self.persons):
            person.take_action(action, self.tax_rate_for_income)

        accumulated_tax = self.get_tax_for_round_for_all()
        self.distribute_tax(accumulated_tax)
        
        # person_rewards = [person.get_reward(is_terminal_state) for person in self.persons]
        # person_next_states = [person.get_state() for person in self.persons]

        # for i, person in enumerate(self.persons):
        #     person.remember(current_states[i], person_actions[i], person_rewards[i], person_next_states[i])
        
        # for person in self.persons:
        #     person.replay()
        
        next_state = self.get_state()

        self.time_step += 1
        return next_state