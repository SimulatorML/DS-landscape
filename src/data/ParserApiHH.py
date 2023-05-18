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
from typing import List, Dict, Tuple, Optional, Set
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
    __MAX_PER_PAGE = 50
    __MAX_VACANCIES_PER_QUERY = 2000

    def __init__(self, config_path: str = SETTINGS_PATH):
        self._config : ParserConfig = config.load(config_path) or ParserConfig()
        self._rates = fetch_exchange_rates()
        self._http_tag_re = re.compile("<.*?>")

        self.ua = UserAgent()
        self.requests_per_session = 0
        self.user_agent = None
        self.session = None
        

    def __random_sleep_api(self) -> None:
        """Sleep for random time preventing blocks from hh"""

        delay_sec = random.randint(self._config.min_random_sleep_ms, self._config.max_random_sleep_ms) / 1000.0
        time.sleep(delay_sec)

    def __random_sleep_http(self) -> None:
        """Sleep for random time preventing blocks from hh"""

        delay_sec = random.randint(self._config.min_random_sleep_ms_http, self._config.max_random_sleep_ms_http) / 1000.0
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
            log.warning('Found maximum possible vacansies for "%s". ' +
                'If you want to get all vacancies reduce period or concretize search.', search_str)

        return ids

    def get_all_vacancies_ids(self, search_requests: List[str], collector: callable) -> pd.DataFrame:
        """Get ids list for all search requests"""

        data = []
        for search_str in search_requests:
            try:
                data.extend([(search_str, x) for x in collector(search_str)])
            except Exception as e:
                log.error('Getting ids failed')
                log.exception(e, stack_info=True)

        df = pd.DataFrame(data, columns=['query', 'vacancy_id'])
        df = df.groupby('vacancy_id')['query'].agg(lambda x: x.to_list()).reset_index()

        log.info(f'Found %s total ids, %s unique', len(data), df.shape[0])
        return df

    def save_vacancies_ids(self, df: pd.DataFrame) -> None:
        """Save ids dataframe to data folder"""

        filename = f"{datetime.today().strftime('%Y-%m-%d')}-IDS.csv"
        filename = os.path.join(self._config.data_path, filename)
        df.to_csv(filename, index=False)

    def load_vacancies_ids(self, filename: str = None) -> pd.DataFrame:
        """Load ids dataframe from data folder
        param: filename: Name of the file. If None filename will be determinated automatically if possible"""

        if filename is None:
            n = 0
            filename = ""
            while not os.path.isfile(filename) and n < 30:
                date = datetime.today() - timedelta(days=n)
                filename = f"{date.strftime('%Y-%m-%d')}-IDS.csv"
                filename = os.path.join(self._config.data_path, filename)
                n += 1

        if os.path.isfile(filename):
            return pd.read_csv(filename)
        else:
            raise ValueError(f'File "{filename}" doesn\'t found')

    def process_ids(self) -> None:
        """Run collection of ids. Result will be saved to data folder with name like '2023-05-17-IDS.txt'"""

        ids = self.get_all_vacancies_ids(self._config.search_requests, self.get_vacancies_ids)
        self.save_vacancies_ids(ids)


    def get_vacancy_from_api(self, vacancy_id: str) -> Vacancy:
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
            description=self._http_tag_re.sub('', row["description"]),
            url=url
        )

        return vacancy

    def _get_http_request(self, url : str) -> requests.Response:
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
    
    def get_vacancy_from_http(self, vacancy_id: str) -> pd.Series:
        """ Get vacancy details via HH HTTP """

        url = f"{self.__HTTP_BASE_URL}{vacancy_id}"
        req = self._get_http_request(url)
        # req = sessia.get(url, headers={'User-Agent': 'Custom'})

        if req is not None and req.status_code == 200:
            soup = BeautifulSoup(req.text, "html.parser")

            uid = re.search('\d+', url).group(0)

            try:
                name = soup.find(['div'], class_='vacancy-title').h1.text
            except:
                name = None

            try:
                salary = soup.find(attrs={'data-qa': 'vacancy-salary'}).text
            except:
                salary = None

            try:
                experience = soup.find(attrs={'data-qa': 'vacancy-experience'}).text
            except:
                experience = None

            try:
                employment_type = soup.find(attrs={'data-qa': 'vacancy-view-employment-mode'}).text
            except:
                employment_type = None

            try:
                company_name = soup.find(['span'], class_='vacancy-company-name').text
            except:
                company_name = None

            try:
                address = soup.find(attrs={'data-qa': 'vacancy-view-raw-address'}).text
            except:
                address = None

            try:
                text = soup.find(['div'], class_='vacancy-description').text
            except:
                text = None

            try:
                skills = str([el.text for el in soup.findAll(attrs={'data-qa': 'bloko-tag__text'})])
            except:
                skills = None

            vacancy = Vacancy(
                vacancy_id=vacancy_id,
                employer=company_name,
                name=name,
                salary_row=salary,
                # salary=salary is not None,
                # salary_from=0,
                # salary_to=0,
                experience=experience,
                schedule=employment_type,
                skills=skills,
                description=text,
                address=address,
                url=url
            )

            return vacancy
            
        else:
            return Vacancy()

    def get_vacancy(self, vacancy_id: str) -> Vacancy:
        """ Get vacancy details via HH HTTP """

        # time.sleep(2.0/1000)
        # return Vacancy(vacancy_id=vacancy_id)

        # return self.get_vacancy_from_api(vacancy_id)
        return self.get_vacancy_from_http(vacancy_id)

    def process_vacancies(self, df : pd.DataFrame, chunk_no : int) -> None:
        """ Process ids to vacancies and save to data folder  """

        filename = f"{datetime.today().strftime('%Y-%m-%d')}-DATA-{chunk_no}.csv"
        filename = os.path.join(self._config.data_path, filename)

        data = []
        for id in tqdm(df['vacancy_id']):
            v = self.get_vacancy(id)
            if v.vacancy_id != '' and v.vacancy_id is not None:
                data.append(v)

        pd.DataFrame(data).to_csv(filename, index=False)

        log.info('Processed chunk %s. Total vacancies %s of %s',
                 chunk_no, len(data), df.shape[0])

    def process_vacancies_chunked(self, ids : pd.DataFrame) -> None:
        """
        Process ids to vacancies and save to data folder
        Process will be splitted to chunks
        """

        chunk_size = self._config.chunk_size
        total_chunks = ids.shape[0] // chunk_size + (1 if ids.shape[0] % chunk_size != 0 else 0)
        for i, df in enumerate([ids.iloc[i:i+chunk_size] for i in range(0, ids.shape[0], chunk_size)]):
            try:
                log.info('Processing chunk %s of %s', i+1, total_chunks)
                self.process_vacancies(df, i+1)
            except Exception as e:
                log.critical('Chunk %s did not processing with exception below:', i+1)
                log.exception(e, stack_info=True)




if __name__ == '__main__':
    if not os.path.isfile(SETTINGS_PATH):
        config.dump(ParserConfig(), SETTINGS_PATH)

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

    # dc.process_ids()

    df = dc.load_vacancies_ids()
    # dc.process_vacancies_chunked(df)



    ids = df
    chunk_size = dc._config.chunk_size
    total_chunks = ids.shape[0] // chunk_size + (1 if ids.shape[0] % chunk_size != 0 else 0)
    for i, df in enumerate([ids.iloc[i:i+chunk_size] for i in range(0, ids.shape[0], chunk_size)]):
        try:
            if i+1 == 5 or i+1 == 11:
                log.info('Processing chunk %s of %s', i+1, total_chunks)
                dc.process_vacancies(df, i+1)
        except Exception as e:
            log.critical('Chunk %s did not processing with exception below:', i+1)
            log.exception(e, stack_info=True)