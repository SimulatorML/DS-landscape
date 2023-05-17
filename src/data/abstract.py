from dataclasses import dataclass
from typing import List

@dataclass
class Vacancy:
    Id : str
    Employer : str
    Name : str
    Salary : bool
    From : int
    To : int
    Experience : str
    Schedule : str
    Keys : List[str]
    Description : str