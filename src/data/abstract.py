from dataclasses import dataclass
from typing import List, Union

@dataclass
class Vacancy:
    vacancy_id : str = None
    employer : str = None
    name : str = None
    salary_row : str = None
    salary : bool = False
    salary_from : Union[int, None] = None
    salary_to : Union[int, None] = None
    experience : str = None
    schedule : str = None
    skills : Union[List[str], None] = None # field(default_factory=lambda: [])
    description : str = None
    address : str = None
    url : str = None
    query : Union[List[str], None] = None # field(default_factory=lambda: [])
