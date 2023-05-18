from dataclasses import dataclass, asdict
import os
from src.data.abstract import Vacancy
from src.utils.logger import configurate_logger
from typing import Union
import pandas as pd


log = configurate_logger('Preprocessor')

@dataclass
class Preprocessor:
    row_data_folder : str
    result_data_folder : str

    def _load_row_file(self, filename: str) -> pd.DataFrame:
        """load one parsed file to dataframe"""
        log.info('loading %s', filename)
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


if __name__ == '__main__':

    preprocessor = Preprocessor(
        row_data_folder='data/hh_parsed_folder', 
        result_data_folder='data/processed')

    df = preprocessor.load_from_folder()
    print(df.shape)
