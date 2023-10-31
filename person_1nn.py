
from person import Person


class Person_1nn(Person):

    def __init__(self, model, idx:int, education_level:float, net_worth:float, base_salary:float = 400.0, epsilon:float=0.1, category:str):
        super().__init__(idx, education_level, net_worth, base_salary, epsilon, category)

        # This is the thing that is different from person.py
        self.model = model