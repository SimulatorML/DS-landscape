from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, asdict, field
import os
import random
import re
import requests
from src.data.abstract import Vacancy
from src.utils import config
from src.utils.logger import configurate_logger
from src.utils.currency_exchange import fetch_exchange_rates
import time
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from urllib.parse import urlencode


SETTINGS_PATH = "cnf/ParserHH.json"
log = configurate_logger('ParserHH')

@dataclass
class ParserConfig:
    min_random_sleep_ms : int = 100
    max_random_sleep_ms : int = 300

    data_path : str = 'data/hh_parsed_folder'
    search_resuests : List[str] = field(default_factory = lambda: (
        ['data scientist', 'аналитик данных', 'machine learning', 'data engineer', 'data analyst']))




class ParserApiHH:
    """Main class for searching vacancies hh.ru via hh api"""
    
    __API_BASE_URL = "https://api.hh.ru/vacancies/"

    def __init__(self, config_path: str = SETTINGS_PATH):
        self._config : ParserConfig = config.load(config_path) or ParserConfig()
        self._rates = fetch_exchange_rates()
        self._http_tag_re = re.compile("<.*?>")

    def __random_sleep(self) -> None:
        """Sleep for random time preventing blocks from hh"""
        delay_sec = random.randint(self._config.min_random_sleep_ms, self._config.max_random_sleep_ms) / 1000.0
        time.sleep(delay_sec)

    def clean_tags(self, html_text: str) -> str:
        return self._http_tag_re.sub("", html_text)

    @staticmethod
    def __convert_gross(is_gross: bool) -> float:
        return 0.87 if is_gross else 1

    def get_vacancy(self, vacancy_id: str):
        # Get data from URL
        url = f"{self.__API_BASE_URL}{vacancy_id}"
        self.__random_sleep()
        vacancy = requests.api.get(url).json()

        # Extract salary
        salary = vacancy.get("salary")

        # Calculate salary:
        # Get salary into {RUB, USD, EUR} with {Gross} parameter and
        # return a new salary in RUB.
        from_to = {"from": None, "to": None}
        if salary:
            is_gross = vacancy["salary"].get("gross")
            for k, v in from_to.items():
                if vacancy["salary"][k] is not None:
                    _value = self.__convert_gross(is_gross)
                    from_to[k] = int(_value * salary[k] / self._rates[salary["currency"]])

        # Create pages tuple
        return (
            vacancy_id,
            vacancy["employer"]["name"],
            vacancy["name"],
            salary is not None,
            from_to["from"],
            from_to["to"],
            vacancy["experience"]["name"],
            vacancy["schedule"]["name"],
            [el["name"] for el in vacancy["key_skills"]],
            self.clean_tags(vacancy["description"]),
        )

    @staticmethod
    def __encode_query_for_url(query: Optional[Dict]) -> str:
        if 'professional_roles' in query:
            query_copy = query.copy()

            roles = '&'.join([f'professional_role={r}' for r in query_copy.pop('professional_roles')])

            return roles + (f'&{urlencode(query_copy)}' if len(query_copy) > 0 else '')

        return urlencode(query)

    def collect_vacancies(self, query: Optional[Dict], debug : bool = False) -> Dict:
        """Parse vacancy JSON: get vacancy name, salary, experience etc.

        Parameters
        ----------
        query : dict
            Search query params for GET requests.

        Returns
        -------
        dict
            Dict of useful arguments from vacancies

        """

        url_params = self.__encode_query_for_url(query)

        # Check number of pages...
        target_url = self.__API_BASE_URL + "?" + url_params
        num_pages = requests.get(target_url).json()["pages"]
        self.__random_sleep()

        if debug:
            num_pages = 1

        # Collect vacancy IDs...
        ids = []
        for idx in range(num_pages + 1):
            response = requests.get(target_url, {"page": idx})
            self.__random_sleep()
            data = response.json()
            if "items" not in data:
                break
            ids.extend(x["id"] for x in data["items"])

        if debug:
            ids = ids[:5]

        # Collect vacancies...
        jobs_list = []
        for vacancy in tqdm(ids):
            jobs_list.append(self.get_vacancy(vacancy))

        return jobs_list









if __name__ == '__main__':
    if not os.path.isfile(SETTINGS_PATH):
        config.dump(ParserConfig(), SETTINGS_PATH)

    dc = ParserApiHH()

    vacancies = dc.collect_vacancies(
        # max per_page = 50
        query={"text": "Data", "area": 113, "per_page": 50, 'period': 20},
        # refresh=True
        debug=True
    )
    print(vacancies)
