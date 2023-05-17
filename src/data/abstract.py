from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class Vacancy:
    vacancy_id : str = ''
    employer : str = ''
    name : str = ''
    salary_row : str = ''
    salary : bool = ''
    salary_from : int = 0
    salary_to : int = 0
    experience : str = ''
    schedule : str = ''
    skills : List[str] = field(default_factory=lambda: [])
    description : str = ''
    address : str = ''
    url : str = ''
