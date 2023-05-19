"""
Proprocessing module (ETL)

Extract: from row_data_folder
Transform: cleaning and creating few simple new features
Load: save to result_data_folder to 'vacancies.csv' file

New features:
    - employment_type - 'Полная занятость', 'Частичная занятость', etc
    - employment_workhours - 'Полный день', 'Гибкий график', etc
    - publish_date - data of publishing
    - city
    - city_rating

Exsample of using:

    preprocessor = Preprocessor(
        row_data_folder='data/hh_parsed_folder',
        result_data_folder='data/processed')

    df = preprocessor.process()

    print(df.shape)
"""

from dataclasses import dataclass, asdict
import os
from src.data.abstract import Vacancy
from src.utils.logger import configurate_logger
from typing import Union
import pandas as pd
from natasha import DatesExtractor, MorphVocab, AddrExtractor
from datetime import date
from pymystem3 import Mystem
from tqdm import tqdm

log = configurate_logger('Preprocessor')

@dataclass
class Preprocessor:
    row_data_folder : str
    result_data_folder : str

    def __post_init__(self):
        tqdm.pandas()

    def _load_row_file(self, filename: str) -> pd.DataFrame:
        """load one parsed file to dataframe"""
        log.info('Loading %s', filename)
        df = pd.read_csv(filename, encoding='utf-8')
        expected_columns = asdict(Vacancy()).keys()
        if not all([x in df.columns for x in expected_columns]):
            raise ValueError(f'Not all colums exists in file {filename}, expected columns {expected_columns}')
        return df

    def load_from_folder(self) -> Union[pd.DataFrame, None]:
        """Load all row data files from directory"""
        df = None
        for fn in [x for x in os.listdir(self.row_data_folder) if '-DATA-' in x and x.endswith('.csv')]:
            filename = os.path.join(self.row_data_folder, fn)
            df_x = self._load_row_file(filename)
            if df is None:
                df = df_x
            else:
                df = pd.concat([df, df_x])

        return df
    
    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """clean df: dropduplicates and some NaN values"""
        df = df.drop_duplicates()
        df = df[df.skills != '[]']
        df = df.dropna(subset=['name', 'description', 'query', 'skills'])
        df = df.reset_index(drop=True)
        df['description'] = df['description'].apply(lambda x: x.strip())
        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Extract features and update current DataFrame, not a copy
            New features:
            - employment_type - 'Полная занятость', 'Частичная занятость', etc
            - employment_workhours - 'Полный день', 'Гибкий график', etc
            - publish_date - data of publishing
            - city
            - city_rating
            """
        
        log.info('Processing employment_type...')
        df['employment_type'] = df.schedule.apply(\
            lambda x: x.split(',')[0].strip() if x is not None and len(x) > 0 else None)

        df['employment_workhours'] = df.schedule.apply(\
            lambda x: x.split(',')[1].strip().capitalize() if x is not None and ',' in x  else None)

        log.info('Processing publish_date...')
        morph_vocab = MorphVocab()
        extractor = DatesExtractor(morph_vocab)
        def extract_publish_date(s):
            if s != s:
                return None
            matches = extractor(s)
            match = next(matches, None)
            return date(match.fact.year, match.fact.month, match.fact.day) if match is not None else None
        df['publish_date'] = df['publish_city_str'].progress_apply(extract_publish_date)

        log.info('Processing city...')
        m = Mystem()
        extractor = AddrExtractor(morph_vocab)
        def extract_city(s):
            if s != s:
                return None
            matches = extractor(s)
            match = next(matches, None)
            return m.lemmatize(match.fact.value)[0].capitalize() if match is not None else None
        df['city'] = df['publish_city_str'].progress_apply(extract_city)
        cities = df['city'].value_counts(normalize=True, dropna=False)
        df['city_rating'] = df.city.apply(lambda x: cities[x])

        return df

    def drop_text_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """ если id-ники вакансий разные, а текст одинаковый, 
             то последенюю по дате публикации, а если даты одинаковые то по более популярному городу
             """
        return df.sort_values(['description', 'publish_date', 'city_rating'],
                                ascending=[True, False, False]). \
                        groupby('description', as_index=False).first()

    def save_df(self, df: pd.DataFrame) -> None:
        """ save resulted dataframe to file"""
        filename = os.path.join(self.result_data_folder, 'vacancies.csv')
        df.to_csv(filename, index=False, encoding='utf-8')
        log.info('Saved to "%s" file', filename)
 
    def process(self) -> pd.DataFrame:
        """
            Extract: from row_data_folder
            Transform: cleaning and creating few simple new features
            Load: save to result_data_folder to 'vacancies.csv' file
            """
        df = self.load_from_folder()
        df = self.data_cleaning(df)
        df = self.extract_features(df)
        df = self.drop_text_duplicates(df)    
        self.save_df(df)
        return df


if __name__ == '__main__':

    preprocessor = Preprocessor(
        row_data_folder='data/hh_parsed_folder',
        result_data_folder='data/processed')

    df = preprocessor.process()

    print(df.shape)