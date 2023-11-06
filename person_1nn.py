
from person import Person


class Person_1nn(Person):

    def __init__(self, model, idx:int, education_level:float, net_worth:float, epsilon:float=0.1, category:str='A'):
        super().__init__(idx, education_level, net_worth, epsilon, category)

        # This is the thing that is different from person.py
        self.model = model