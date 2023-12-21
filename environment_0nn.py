import random
from concurrent.futures import ThreadPoolExecutor

import configuration
import utils
from bid_sell import *
from environment import Environment
from person_0nn import Person_0nn
from policyplanneragent import PolicyPlannerAgent

executor = ThreadPoolExecutor()  # You can adjust the number of workers as appropriate.


def select_action_for_person(person, time_step, horizon, tax_function):
    """
    Args:
        person: Person agent
        time_step: current step in lifecycle
        horizon: how many rounds to the future the person sees
        tax_function: the perceived tax function


    Returns: action of person
    """
    return person.select_action(time_step, horizon, tax_function)


class Environment_0nn(Environment):
    def __init__(self, n_persons: int, horizon: int, random_seed=1):
        super().__init__(n_persons, random_seed=random_seed)
        self.available_category_of_person = ["A", "B", "C"]

        # For person decisions
        self.horizon = horizon
        self.time_step = 0
        self.discount_rate_func = utils.get_discount_rate_heuristic(
            configuration.config.get_constant("DISCOUNT_RATE_HEURISTIC"))

        random.seed(self.random_seed)
        self.persons = [Person_0nn(idx, random.choice(self.education_level_turn0),
                                   self.net_worth_turn0, category=random.choice(
                self.available_category_of_person),
                                   discount_rate=self.discount_rate_func()) for idx in
                        range(n_persons)]

        self.PolicyPlannerAgent = PolicyPlannerAgent(2 * n_persons + self.n_brackets,
                                                     len(configuration.config.get_constant(
                                                         "ACTIONS")))

        self.bid_sell_system = BidSellSystem(
            commodities=self.available_category_of_person, agents=self.persons)

    def persons_step(self, is_terminal_state=False):
        """
        takes a step for each Person in the system

        Args:
            is_terminal_state: if the last step in current life-cycle

        Returns:

        """

        # for concurrent execution - in my PC takes longer than reg run
        # person_actions = self.get_person_actions_concurrently()

        person_actions = [person.select_action(self.time_step, self.horizon,
                                               self.PolicyPlannerAgent.tax_rate_for_income)
                          for person in self.persons]

        for action, person in zip(person_actions, self.persons):
            person.take_action(action, self.PolicyPlannerAgent.tax_rate_for_income)

        accumulated_tax = self.get_tax_for_round_for_all()
        self.distribute_tax(accumulated_tax)

        next_state = self.get_state()

        self.time_step += 1
        return next_state

    def reset_persons(self, reset_seed: bool = True):
        """
        resets Persons after each life-cycle

        Args:
            reset_seed: resets the seed so in setting of Persons the same choice sare made
                        like previous life-cycles.
        """

        if reset_seed:
            random.seed(self.random_seed)
        self.persons = [Person_0nn(idx, random.choice(self.education_level_turn0),
                                   self.net_worth_turn0,
                                   category=random.choice(["A", "B", "C"]),
                                   discount_rate=self.discount_rate_func()) for idx in
                        range(len(self.persons))]

    def reset(self):
        """
        resets the enviroment after each life-cycle
        """

        self.time_step = 0

    def get_person_actions_concurrently(self):
        """
        selects actions for all persons concurrently

        Returns: a list of actions for persons

        """
        # Submit tasks to the already created pool executor.
        futures = [executor.submit(select_action_for_person, person, self.time_step,
                                   self.horizon,
                                   self.PolicyPlannerAgent.tax_rate_for_income) for person
                   in self.persons]

        # Wait for all the tasks to complete and gather the results.
        person_actions = [future.result() for future in futures]
        return person_actions

    # override functions
    def persons_gain_category_token(self):
        pass

    def persons_do_bid_sell(self):
        pass

    def update_history_of_auctions(self):
        pass

    def remove_redundant_current_dict(self):
        pass
