
EDUCATION_LEVELS = [0, 1, 2, 3, 4]
EDUCATION_EARNINGS = {0: 10000, 1: 13000, 2: 15000, 3: 17000, 4: 20000}
EXPENSE = 10000
NUM_PERSONS = 100
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
BATCH_SIZE = 64
MEMORY_SIZE = 10000
ACTIONS = [0, 1, 2]  # or any other actions
SALARIES = [10000, 13000, 16000, 19000, 22000]
EDUCATION_INCREASE = 0.1




class PolicyPlanner:

    def choose_action(self,dawmkdwa)
        
        raise

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)


