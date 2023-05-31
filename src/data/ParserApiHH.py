"""
Parser vacancies from hh.ru

Settings here SETTINGS_PATH = 'cnf/ParserApiHH.json'
        avaliable options and default values in ParserConfig class

Exsample of using:

    # create object
    dc = ParserApiHH()

    # FIRST stage (sometimes vpn is needed)
    parsed_ids = dc.get_parsed_ids()
    dc.process_ids(parsed_ids)

    # SECOND stage
    df, filename = dc.load_vacancies_ids()
    dc.process_vacancies_chunked(df, filename)

    All data will be stored in data folder, 'data/hh_parsed_folder' by default        

"""

from bs4 import BeautifulSoup
from dataclasses import dataclass, fields, asdict, field
from datetime import datetime, timedelta
from fake_useragent import UserAgent
import os
import random
import re
import requests
from src.data.abstract import Vacancy
from src.utils import config
from src.utils.logger import configurate_logger
from src.utils.currency_exchange import fetch_exchange_rates
import time
from typing import List, Dict, Tuple, Optional, Set, Union
from tqdm import tqdm
from urllib.parse import urlencode
import pandas as pd

SETTINGS_PATH = "cnf/ParserApiHH.json"
log = configurate_logger('ParserHH')

@dataclass
class ParserConfig:
    # via hh api
    min_random_sleep_ms : int = 100
    max_random_sleep_ms : int = 300

    # via hh http
    min_random_sleep_ms_http : int = 1500
    max_random_sleep_ms_http : int = 4000
    max_requests_per_session : int = 100   

    # vacancies processing
    chunk_size : int = 500

    # 113 - Russia
    # 1 - Moskow
    # https://api.hh.ru/areas
    area : int = 113
    period_days : int = 2

    data_path : str = 'data/hh_parsed_folder'
    search_requests : List[str] = field(default_factory = lambda: (
        ['data scientist', 'аналитик данных', 'machine learning', 'data engineer', 'data analyst']))

class ParserApiHH:
    """Main class for searching vacancies hh.ru via hh api"""

    __API_BASE_URL = "https://api.hh.ru/vacancies/"
    __HTTP_BASE_URL = "https://hh.ru/vacancy/"
    __MAX_PER_PAGE = 100
    __MAX_VACANCIES_PER_QUERY = 2000
    __DEFAULT_TAX = 0.13

    def __init__(self, config_path: str = SETTINGS_PATH):
        self._config : ParserConfig = config.load(config_path) or ParserConfig()
        
        self._rates = fetch_exchange_rates()

        self._http_tag_re = re.compile("<.*?>")
        self._number_re = re.compile(r'\d+')

        self.ua = UserAgent()
        self.requests_per_session = 0
        self.user_agent = None
        self.session = None

    def __random_sleep_api(self) -> None:
        """Sleep for random time preventing blocks from hh"""

        delay_sec = random.randint(self._config.min_random_sleep_ms,
                                   self._config.max_random_sleep_ms) / 1000.0
        time.sleep(delay_sec)

    def __random_sleep_http(self) -> None:
        """Sleep for random time preventing blocks from hh"""

        delay_sec = random.randint(self._config.min_random_sleep_ms_http,
                                   self._config.max_random_sleep_ms_http) / 1000.0
        time.sleep(delay_sec)

    def fetch_request(self, target_url: str, params = None) -> requests.Response:
        """Fetch hh request and sleep random delay"""

        response = requests.get(target_url, params, timeout=5000)
        self.__random_sleep_api()
        return response

    def get_vacancies_ids(self, search_str: str) -> List[str]:
        """Get ids list for one search_str request"""

        query = {
            'text': search_str, 
            'area': self._config.area,
            'per_page' : self.__MAX_PER_PAGE,
            'period' : self._config.period_days
        }

        log.info('Getting ids for "%s"', search_str)

        target_url = self.__API_BASE_URL + "?" + urlencode(query)
        response = self.fetch_request(target_url)
        num_pages = response.json()["pages"]

        ids = []
        for idx in range(num_pages + 1):
            response = self.fetch_request(target_url, {"page": idx}) if idx != 0 else response
            data = response.json()
            if "items" not in data:
                break
            ids.extend(x["id"] for x in data["items"])

        log.info('Found %s ids', len(ids))
        if len(ids) == self.__MAX_VACANCIES_PER_QUERY:
            log.warning('Found maximum possible vacansies %s for "%s"',
                        self.__MAX_VACANCIES_PER_QUERY, search_str)

        return ids

    def get_all_vacancies_ids(self,
                              search_requests: List[str],
                              collector: callable,
                              parsed_ids: Set[str] = None) -> pd.DataFrame:
        """Get ids list for all search requests
        
        param: search_requests: list of rsearch requests
        param: collector: callable(search_str: str) -> List[str] return ids from one search request
                            the collector moved to param for simplify unit testing
        param: parsed_ids: set of ignored ids
                            """

        if parsed_ids is None:
            parsed_ids = set([])

        data = []
        for search_str in [x for x in search_requests if not x.startswith('--')]:
            try:
                data.extend([(search_str, x) for x in collector(search_str)])
            except Exception as e:
                log.error('Getting ids failed')
                log.exception(e, stack_info=True)

        df = pd.DataFrame(data, columns=['query', 'vacancy_id'])
        df = df.groupby('vacancy_id')['query'].agg(lambda x: x.to_list()).reset_index()
        unique_cnt = df.shape[0]

        df = df[~df['vacancy_id'].isin(parsed_ids)]

        log.info(f'Found %s total ids, %s unique, %s new', len(data), unique_cnt, df.shape[0])
        return df

    def save_vacancies_ids(self, df: pd.DataFrame) -> None:
        """Save ids dataframe to data folder"""

        filename = f"{datetime.today().strftime('%Y-%m-%d')}-IDS.csv"
        filename = os.path.join(self._config.data_path, filename)
        df.to_csv(filename, index=False)

    def load_vacancies_ids(self, filename: str = None) -> Tuple[pd.DataFrame, str]:
        """Load ids dataframe from data folder
        param: filename: Name of the file. If None filename will be determinated automatically if possible
        return: Tuple[DataFrame, filename]"""

        if filename is None:
            n = 0
            filename = ""
            while not os.path.isfile(filename) and n < 30:
                date = datetime.today() - timedelta(days=n)
                filename = f"{date.strftime('%Y-%m-%d')}-IDS.csv"
                filename = os.path.join(self._config.data_path, filename)
                n += 1

        if os.path.isfile(filename):
            return (pd.read_csv(filename), filename)
        else:
            raise ValueError(f'File "{filename}" doesn\'t found')

    def process_ids(self, parsed_ids : Set[str] = None) -> None:
        """Run collection of ids. Result will be saved to data folder with name like '2023-05-17-IDS.txt'
        param: process_ids: set of ignored ids"""

        ids = self.get_all_vacancies_ids(
            self._config.search_requests, self.get_vacancies_ids, parsed_ids)
        self.save_vacancies_ids(ids)

    def get_vacancy_from_api(self, vacancy_id: str, query: str) -> Union[Vacancy, None]:
        """ Get vacancy details via HH API """

        url = f"{self.__API_BASE_URL}{vacancy_id}"
        self.__random_sleep_api()
        row = requests.api.get(url, timeout=5000).json()

        salary = row.get("salary")
        from_to = {"from": None, "to": None}
        if salary:
            is_gross = row["salary"].get("gross")
            for k, v in from_to.items():
                if row["salary"][k] is not None:
                    gross_koef = 0.87 if is_gross else 1
                    from_to[k] = int(gross_koef * salary[k] / self._rates[salary["currency"]])

        vacancy = Vacancy(
            vacancy_id=vacancy_id,
            employer=row["employer"]["name"],
            name=row["name"],
            salary=salary is not None,
            salary_from=from_to["from"],
            salary_to=from_to["to"],
            experience=row["experience"]["name"],
            schedule=row["schedule"]["name"],
            skills=[el["name"] for el in row["key_skills"]],
            description=self._http_tag_re.sub('', row["description"]).replace('\n', ' '),
            url=url,
            query=query
        )

        return vacancy

    def _get_http_request(self, url : str) -> Union[requests.Response, None]:
        """Do request with fake user_agent and session control"""
        self.requests_per_session += 1
        if self.session is None or self.requests_per_session % self._config.max_requests_per_session == 0:
            self.user_agent = self.ua.random
            self.requests_per_session = 0
            self.session = requests.Session()

        try:
            req = self.session.get(url, headers={'User-Agent': self.user_agent})
            self.__random_sleep_http()
        except requests.exceptions.RequestException as e:
            self.session = None
            log.warning('http session crashed with error: %s', e)
            return None

        if req.status_code != 200:
            self.session = None
            log.warning('request return status_code: %s (%s)', req.status_code, url)
            return req

        return req

    def _parse_salary(self, salary_row) -> Tuple[bool, int, int]:
        """ Parse salary string
        return: has_salary, salary_rom, salary_to"""

        salary_row = salary_row or ''

        s = salary_row.replace(' ', '').replace(u'\xa0', '')
        if s is None or s == '':
            return False, None, None

        try:
            match = re.findall(self._number_re, s)
            if len(match) == 0:
                return False, None, None

            exchange_rate = 1.0
            for k, v in self._rates.items():
                if k in s:
                    exchange_rate = v
                    break

            tax = self.__DEFAULT_TAX if s.lower().find('до вычета налогов'.replace(' ', '')) > 0 else 0
            koeff = (1 - tax) / exchange_rate

            if len(match) == 1:
                return True, int(koeff * int(match[0])), int(koeff * int(match[0]))
            else:
                return True, int(koeff * int(match[0])), int(koeff * int(match[1]))

        except Exception as e:
            log.warning('Salary parsing exception salary_row="%s", message="%s"', salary_row, e.args)
            return False, None, None

    def get_vacancy_from_http(self, vacancy_id: str, query: str) -> Union[Vacancy, None]:
        """ Get vacancy details via HH HTTP """

        url = f"{self.__HTTP_BASE_URL}{vacancy_id}"
        req = self._get_http_request(url)

        if req is None or req.status_code != 200:
            return None

        soup = BeautifulSoup(req.text, "html.parser")
        # uid = re.search('\d+', url).group(0)

        def get_text(x):
            return x.text if x is not None else None

        n = soup.find(['div'], class_='vacancy-title')
        name = get_text(n.h1) if n is not None else None

        salary_row = get_text(soup.find(attrs={'data-qa': 'vacancy-salary'}))
        experience = get_text(soup.find(attrs={'data-qa': 'vacancy-experience'}))
        employment_type = get_text(soup.find(attrs={'data-qa': 'vacancy-view-employment-mode'}))
        company_name = get_text(soup.find(['span'], class_='vacancy-company-name'))
        address = get_text(soup.find(attrs={'data-qa': 'vacancy-view-raw-address'}))
        text = get_text(soup.find(['div'], class_='vacancy-description'))
        publish_city_str = get_text(soup.find(['p'], class_='vacancy-creation-time-redesigned'))

        s = soup.findAll(attrs={'data-qa': 'bloko-tag__text'})
        skills = str([el.text for el in s]) if s is not None else []

        salary, salary_from, salary_to = self._parse_salary(salary_row)

        vacancy = Vacancy(
            vacancy_id=vacancy_id,
            employer = company_name.replace(u'\xa0', ' ')
                        if company_name is not None else None,
            name=name,
            salary_row=salary_row,
            salary=salary,
            salary_from=salary_from,
            salary_to=salary_to,
            experience=experience,
            schedule=employment_type,
            skills=skills,
            description = text.replace('\n', ' ').replace(u'\xa0', ' ')
                            if text is not None else None,
            address=address,
            url=url,
            query=query,
            publish_city_str = publish_city_str.replace('\n', ' ').replace(u'\xa0', ' ')
                                if publish_city_str is not None else None,
        )

        return vacancy


    def get_vacancy(self, vacancy_id: str, query: str) -> Union[Vacancy, None]:
        """ Get vacancy details via HH HTTP """

        # time.sleep(2.0/1000)
        # return Vacancy(vacancy_id=vacancy_id)

        # return self.get_vacancy_from_api(vacancy_id, query)
        return self.get_vacancy_from_http(vacancy_id, query)

    def process_vacancies(self, df : pd.DataFrame, chunk_no : int, filename: str) -> None:
        """ Process ids to vacancies and save to data folder  """

        data = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            v = self.get_vacancy(row['vacancy_id'], row['query'])
            if v is not None:
                data.append(v)

        pd.DataFrame(data).to_csv(filename, index=False, encoding='utf-8')

        log.info('Processed chunk %s. Total vacancies %s of %s',
                 chunk_no, len(data), df.shape[0])

    def process_vacancies_chunked(self, ids : pd.DataFrame, id_filename: str,
                                  specify_chunks : List[int] = None) -> None:
        """
        Process ids to vacancies and save to data folder
        Process will be splitted to chunks
        """

        id_file_ending = 'IDS.csv'
        if not id_filename.endswith(id_file_ending):
            raise ValueError(f'filename if id_file must be ended with "{id_file_ending}"')

        chunk_size = self._config.chunk_size
        total_chunks = ids.shape[0] // chunk_size + (1 if ids.shape[0] % chunk_size != 0 else 0)
        for i, df in enumerate([ids.iloc[i:i+chunk_size] for i in range(0, ids.shape[0], chunk_size)]):
            try:
                chunk_no = i+1
                if specify_chunks is None or chunk_no in specify_chunks:
                    log.info('Processing chunk %s of %s', chunk_no, total_chunks)
                    filename = id_filename.replace(id_file_ending, f'DATA-{chunk_no}.csv')
                    self.process_vacancies(df, chunk_no, filename)
            except Exception as e:
                log.critical('Chunk %s did not processing with exception below:', chunk_no)
                log.exception(e, stack_info=True)

    def get_parsed_ids(self) -> Set[str]:
        """
        Find all parsed vacancy ids
        It's just collecting all saved data inside data folder
        In future it must be 
        """

        ids = set([])
        for fn in [x for x in os.listdir(self._config.data_path) if '-DATA-' in x and x.endswith('.csv')]:
            df = pd.read_csv(os.path.join(self._config.data_path, fn))
            ids = ids.union(list(df['vacancy_id'].astype(str)))

        return ids


if __name__ == '__main__':
    if not os.path.isfile(SETTINGS_PATH):
        config.dump(ParserConfig(), SETTINGS_PATH)

    # useful for unit tests

    dc = ParserApiHH()

    # vacancies = dc.collect_vacancies(
    #     # max per_page = 50
    #     query={"text": "Data", "area": 113, "per_page": 50, 'period': 20},
    #     # refresh=True
    #     debug=True
    # )
    # print(vacancies)

    #x = dc.get_vacancies_ids('ml')
    
    #dc.save_vacancies_ids(set(["11", "12", "22"]))

    # list = [('a', '1'), ('a', '2'), ('b', '1')]
    # df = pd.DataFrame(list, columns=['col1', 'col2'])
    # df = df.groupby('col2')['col1'].agg(lambda x: [y for y in x]).reset_index()
    # dc.save_vacancies_ids(df)

    # !!!! this is a FIRST stage
    parsed_ids = dc.get_parsed_ids()
    dc.process_ids(parsed_ids)

    # !!!! this is a SECOND stage
    df, filename = dc.load_vacancies_ids()
    dc.process_vacancies_chunked(df, filename, specify_chunks=None)

    #dc.process_vacancies(df.head(10), 1, 'temp/data.csv')

    #print(filename)
    #print(df.shape)

    #parsed_ids = dc.get_parsed_ids()
    #print(len(parsed_ids))

