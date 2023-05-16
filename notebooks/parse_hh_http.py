import sys
from typing import List, Dict, Tuple
import getopt
import os
import random
import re
import requests
import time

import csv
from bs4 import BeautifulSoup
from dataclasses import dataclass, fields, asdict, field
from fake_useragent import UserAgent
import json
import jsonpickle
import logging
from logging.handlers import RotatingFileHandler

@dataclass
class Vacancy:
    name : str
    salary : str
    experience : str
    employment_type : str
    company_name : str
    adress : str
    text : str
    skills : str
    uid : str

@dataclass
class ParserConfig:
    min_random_sleep_ms : int = 1500
    max_random_sleep_ms : int = 4000
    max_requests_per_session : int = 100
    data_path : str = 'data/hh_parsed_folder'
    search_resuests : List[str] = field(default_factory = lambda: (
        ['data scientist', 'аналитик данных', 'machine learning', 'data engineer', 'data analyst']))


# def parse_page(url, sessia):
#     """
#     Parsing Page, get params from html and convert to dict
#     :param url:
#     :param sessia:
#     :return: dict
#     """
#     #print('hello world')
#     req = sessia.get(url, headers={'User-Agent': 'Custom'})
#     if req.status_code == requests.codes.ok:
#         soup = BeautifulSoup(req.text, "html.parser")

#         uid = re.search('\d+', url).group(0)

#         try:
#             name = soup.find(['div'], class_='vacancy-title').h1.text
#         except:
#             name = None

#         try:
#             salary = soup.find(attrs={'data-qa': 'vacancy-salary'}).text
#         except:
#             salary = None

#         try:
#             experience = soup.find(attrs={'data-qa': 'vacancy-experience'}).text
#         except:
#             experience = None

#         try:
#             employment_type = soup.find(attrs={'data-qa': 'vacancy-view-employment-mode'}).text
#         except:
#             employment_type = None

#         try:
#             company_name = soup.find(['span'], class_='vacancy-company-name').text
#         except:
#             company_name = None

#         try:
#             adress = soup.find(attrs={'data-qa': 'vacancy-view-raw-address'}).text
#         except:
#             adress = None

#         try:
#             text = soup.find(['div'], class_='vacancy-description').text
#         except:
#             text = None

#         try:
#             skills = str([el.text for el in soup.findAll(attrs={'data-qa': 'bloko-tag__text'})])
#         except:
#             skills = None

#         parse_params = {'name': name,
#                         'salary': salary,
#                         'experience': experience,
#                         'employment_type': employment_type,
#                         'company_name': company_name,
#                         'adress': adress,
#                         'text': text,
#                         'skills': skills,
#                         'uid': uid
#                         }

#         return parse_params

class hh_parser:

    def __init__(self, config_file_name : str = None) -> None:
        
        jsonpickle.set_preferred_backend('json')
        jsonpickle.set_encoder_options('json', ensure_ascii=False)

        if config_file_name is not None and os.path.isfile(config_file_name):
            with open(config_file_name, 'r', encoding='utf-8') as f:
                self.config = jsonpickle.decode(f.read())
            if not isinstance(self.config, ParserConfig):
                raise ValueError(f'Invalid parser config. You can make default config by make_default_config method')
        else:
            self.config = ParserConfig()

        self.log = logging.getLogger(__name__)
        self.ua = UserAgent()
        self.requests_per_session = 0
        self.user_agent = None
        self.session = None

    @staticmethod
    def make_default_config(config_file_name : str = 'cnf/hh_parser.json') -> None:
        """Save file with default config"""
        jsonpickle.set_preferred_backend('json')
        jsonpickle.set_encoder_options('json', ensure_ascii=False)

        with open(config_file_name, 'w', encoding='utf-8') as f:
            f.write(jsonpickle.encode(ParserConfig()))

    def _random_sleep(self) -> None:
        """Sleep for random time preventing blocks from hh"""
        delay_sec = random.randint(self.config.min_random_sleep_ms, self.config.max_random_sleep_ms) / 1000.0
        time.sleep(delay_sec)
    
    def _get_request(self, url : str, payload: Dict[str, any] = None) -> requests.Response:
        """Do request with fake user_agent and session control"""
        self.requests_per_session += 1
        if self.session is None: #! or self.requests_per_session % self.config.max_requests_per_session == 0:
            self.user_agent = 'Custom' #! self.ua.random
            self.requests_per_session = 0
            self.session = requests.Session()

        try:
            req = self.session.get(url,
                            headers={'User-Agent': self.user_agent},
                            data=payload)
            self._random_sleep()
        except requests.exceptions.RequestException as e:
            self.session = None
            self.log.warning('session crashed with error: %s', e)
            return None

        if req.status_code != 200:
            self.session = None
            self.log.warning('session stopped with status_code: %s', req.status_code)
            return None

        return req


    def get_links_of_vacancies(self,
            search_str : str,
            search_period_days : int = 14,
            area : int = 113,
            pages_limit : int = None
        ) -> List[Tuple[str, str]]:
        """
        Get list of url of vacancies from hh.ru

        :param search_str: Vacancy name which one will be parsed
        :param search_period: Filtred last n-days, default = 14 (2week)
        :param area: Area code, default = 113 (Russia)
        :pages_limit: Limit pages count
        :return: List of Typle[name, url]
        """

        re_vacancy_id = re.compile(r'vacancyId=[0-9]+')
        result = []
        page_number = 0
        while pages_limit is None or page_number < pages_limit:
            payload = {'search_period': search_period_days,
                        'text': search_str,
                        'ored_clusters': True,
                        'enable_snippets': True,
                        'clusters': True,
                        'area': area,
                        'hhtmFrom': 'vacancy_search_catalog',
                        'page': page_number,
                        }
            
            page_number += 1
            #req = self._get_request('https://hh.ru/search/vacancy', payload)
            req = self._get_request(f'https://hh.ru/search/vacancy?text=data+scientist&page={page_number}')
            if req is None:
                break


            with open('temp/temp.html', 'w') as f:
                f.write(req.text)

            self.log.info('[PAGE] %s - %s', page_number, search_str)
            soup = BeautifulSoup(req.text, "html.parser")
            matches = soup.findAll(['div'], class_='vacancy-serp-item__layout')
            if len(matches) == 0:
                self.log.info('No more vacancies on the page')
                break

            for el in matches:
                a_ref = el.find('a', class_='serp-item__title', href=True)

                if a_ref is not None and a_ref.has_key('href') and 'vacancyId=' in a_ref["href"]:

                    # '/applicant/vacancy_response?vacancyId=80316762&hhtmFrom=vacancy_search_list'
                    # https://hh.ru/vacancy/80316762
                    # vacancy_id = re_vacancy_id.findall(a_ref["href"])[0].replace('vacancyId=', '')
                    url = str(a_ref["href"]).split('?')[0] # f'https://hh.ru/vacancy/{vacancy_id}'
                    _name = a_ref.text
                    self.log.debug('%s - %s', url, _name)
                    result.append((_name, url))

        return result

    def save_links_of_vacancies(self, links : List[str]) -> None:
        """Save list to data folder"""

        with open(f'{self.config.data_path}/links.txt', 'w', encoding='utf-8') as f:
            for s in links:
                f.write(s + '/n')

    def load_links_of_vacancies(self) -> List[str]:
        """Load list from cache folder"""

        with open(f'{self.config.data_path}/links.txt', 'r', encoding='utf-8') as f:
            return f.readlines()

    def get_vacancies(self, links : List[str]) -> List[Vacancy]:
        """
        Get vacancies list from hh.ru

        :param search_str: Vacancy name which one will be parsed
        :param search_period: Filtred last n-days, default = 14 (2week)
        :param area: Area code, default = 113 (Russia)
        :pages_limit: Limit pages count
        :return: List of vacancy
        """

        pass

    def save_vacancies(self, vacancies : List[Vacancy], save_path='data/hh_parsed_folder') -> str:
        """
        :param vacancies: List of dictionary
        :param save_path: Path for saving file
        :return: saved file name

        encoded in utf-8
        """

        if not os.path.exists(save_path):
            self.log.critical('Save path doesn\'t exists: "%s", saved in current directory instead')
            save_path = '.'

        if len(vacancies) == 0:
            return

        file_name = f"{save_path}/hh_tmp.csv"
        with open(file_name, "w", encoding='utf-8') as f:
            keys = [fld.name for fld in fields(Vacancy)]
            csvwriter = csv.DictWriter(f, keys)
            csvwriter.writeheader()
            csvwriter.writerows([asdict(prop) for prop in vacancies])

        return file_name

def configurate_logger():
    """Configure logging
        command-line arguments
        --log=DEBUG or --log=debug
        --logfile=hh_parser.log
    """
    log_level = logging.DEBUG
    logfile = 'hh_parser.log'
    for argument, value in getopt.getopt(sys.argv[1:], [], ['log=', 'logfile='])[0]:
        if argument in {'--log'}:
            log_level = getattr(logging, value.upper(), None)
            if not isinstance(log_level, int):
                raise ValueError(f'Invalid log level: {value}')
        elif argument in {'--logfile'}:
            logfile = value

    format_str = '%(asctime)s  %(levelname)s:  %(message)s'

    log = logging.getLogger(__name__)
    log.setLevel(log_level)

    formatter = logging.Formatter(format_str)
    handler = RotatingFileHandler(logfile, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    log.addHandler(handler)

    formatter = logging.Formatter(format_str)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log.addHandler(handler)


if __name__ == '__main__':

    configurate_logger()
    log = logging.getLogger(__name__)

    #search_vacancy_list = ['data scientist', 'аналитик данных', 'machine learning', 'data engineer', 'data analyst']
    search_vacancy_list = ['data scientist']

    for s in search_vacancy_list:
        try:
            parser = hh_parser()
            links = parser.get_links_of_vacancies(s, pages_limit=3)
            #vacancies = get_data(s, pages_limit=2)
            #save_vacancies(vacancies)
        except Exception as e:
            log.exception(e, stack_info=True)
