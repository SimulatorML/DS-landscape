"""
Feature processing:
    Default:
        data_folder='data/processed'
        features_folder='data/features'
        config_folder='cnf'

    Config (config_folder):
        - 'professions.json': professions json

    Input (data_folder):
        - 'vacancies.csv': Dataframe

    Output (features_folder):
        - 'skills.txt': all skills after corrections ordered by name
        - 'skill_index_to_corrected.pkl': dictionary skill index to corrected skill name
        - 'skill_original_to_index.pkl': dictionary skill name to index
        - 'skills.csv': dataframe with columns:
                ['skill_name', 'skill_id', 'salary_q25', 'salary_q50', 'salary_q75', 'frequency', 
                    'popular_profession_id', 'popular_profession_name', <professions>]
        - 'prof.csv': dataframe with columns:
                ['prof_name', 'prof_id', 'salary_q25', 'salary_q50', 'salary_q75', 'frequency', 'popular_skills']
        - 'matrix.csv': skill-profession relationship matrix (numpy)

        Save matrix to 'matrix.csv' file

Exsample of using:
    FeaturesProcessor().process()


"""

import os
import numpy as np
import pandas as pd
from src.utils import config
from src.utils.logger import configurate_logger
from typing import Tuple, Dict, List
import re
from tqdm import tqdm
import pickle

import warnings
warnings.filterwarnings("ignore")

log = configurate_logger('FeaturesProcessor')

class FeaturesProcessor:

    def __init__(self, 
                data_folder='data/processed',
                features_folder='data/features',
                config_folder='cnf',
                min_vacancies_for_skill = 10):

        tqdm.pandas()

        # vacancies data frame
        filename = os.path.join(data_folder, 'vacancies.csv')
        if not os.path.isfile(filename):
            raise ValueError(f'File does not exists: {filename}')
        self.df = pd.read_csv(filename, encoding='utf-8')

        # professions
        filename = os.path.join(config_folder, 'professions.json')
        if not os.path.isfile(filename):
            raise ValueError(f'File does not exists: {filename}')
        professions_map : List[Dict[str, str]] = config.load(filename)
        self.professions = pd.DataFrame(columns=['query_name', 'profession'])
        for row in professions_map:
            self.professions = self.professions.append(row, ignore_index=True)

        # skill aliases
        filename = os.path.join(config_folder, 'skill_aliases.json')
        if not os.path.isfile(filename):
            raise ValueError(f'File does not exists: {filename}')
        self.skill_aliases : List[List[str]] = config.load(filename)

        self.features_folder = features_folder
        self.min_vacancies_for_skill = min_vacancies_for_skill


    def _extract_skills(self) -> Dict[str, float]:
        """
            Add skill_set column to self.df that contain set of skills for row
            Returns sill dictionary

            return: Dict[str, float]:
                Skills dictionary. Key is skill. Value is normalized frequency
        """
        log.info('Extracting skills...')
        r = re.compile('\\\\|////|,')
        self.df['skill_set'] = self.df['skills'].apply(
            lambda s : {x.strip(" '") for x in re.split(r, s.strip('[]'))} - {''})
        all_skills = set().union(*self.df.skill_set.to_list())

        skills = {}
        rows_count = self.df.shape[0]
        for skill in tqdm(all_skills):
            skills[skill] = self.df.skill_set.apply(lambda x: skill in x).sum() / rows_count

        return skills

    def _skills_corrections(self, skills: Dict[str, float]) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Skills correction:
            - discarding case, spec-sumbols
            - <some more later... (rule-based, spell-shecking, etc)>
            - choosing the best skill name based frequency of skills in group

            Parameters
            ----------
            skills: Dict[str, float] :
                Dict skill name to frequency

            Returns
            -------
            Tuple[Dict[int, str], Dict[str, int]] :
                Tuple of:
                    - dist if original skill name to index
                    - dict index to corrected skill name
        
        """

        log.info('Correcting skills...')

        def simplify(s):
            for x in self.skill_aliases:
                if s.lower() in [y.lower() for y in x]:
                    s = x[0]
                    break

            return s.lower().replace(' ', '').replace('-', '').replace('/','').replace(':','')

        new_index  = 0
        index_to_corrected = {}
        original_to_index = {}
        simplified_to_index = {}
        for skill in skills.keys():
            simplified_skill = simplify(skill)
            index = simplified_to_index.get(simplified_skill, new_index)
            original_to_index[skill] = index

            if index == new_index:
                simplified_to_index[simplified_skill] = index
                index_to_corrected[index] = skill
                new_index += 1
            else:
                skill_aliases = [k for k, v in original_to_index.items() if v == index]
                if skill not in skill_aliases:
                    skill_aliases += [skill]
                best_name = skill_aliases[np.array([skills[x] for x in skill_aliases]).argmax()]
                index_to_corrected[index] = best_name

        return original_to_index, index_to_corrected

    def _create_skill_df(self, original_to_index: Dict[str, int],
                        index_to_corrected: Dict[int, str]) -> pd.DataFrame:
        """
            Create skills dataframe and calculate skills features.
            Dataframe columns:
                - skill_name: skill name
                - skill_id: id
                - salary_q25: 25 salary quantile 
                - salary_q50: 50 salary quantile 
                - salary_q75: 75 salary quantile 
                - frequency: relative frequency
        """

        log.info('Creating skills data frame...')

        skill_df = pd.DataFrame()
        skill_df['skill_name'] = index_to_corrected.values()
        skill_df['skill_id'] = index_to_corrected.keys()

        salary_q25 = {}
        salary_q50 = {}
        salary_q75 = {}
        skill_freq = {}

        df = self.df[['skill_set', 'salary', 'salary_from', 'salary_to']].copy()
        df['skill_ind_set'] = df['skill_set'].apply(lambda x: set([original_to_index[s] for s in x]))

        for s in skill_df.skill_id:
            s_df = df[df['skill_ind_set'].apply(lambda x: s in x)]
            skill_freq[s] = s_df.shape[0] / df.shape[0]
            s_df = s_df[s_df['salary']]
            if s_df.shape[0] > self.min_vacancies_for_skill:
                salaries = (s_df['salary_from'] + s_df['salary_to']) / 2
                salary_q25[s] = int(salaries.quantile(0.25))
                salary_q50[s] = int(salaries.quantile(0.50))
                salary_q75[s] = int(salaries.quantile(0.75))

        skill_df['salary_q25'] = skill_df['skill_id'].apply(lambda x: salary_q25.get(x, None))
        skill_df['salary_q50'] = skill_df['skill_id'].apply(lambda x: salary_q50.get(x, None))
        skill_df['salary_q75'] = skill_df['skill_id'].apply(lambda x: salary_q75.get(x, None))
        skill_df['frequency'] = skill_df['skill_id'].apply(lambda x: skill_freq.get(x, None))

        log.info('Created skills data frame')

        return skill_df


    def save_skill_df(self) -> None:
        """Save skill dataframe to file
        'skills.csv': dataframe with columns:
                ['skill_name', 'skill_id', 'salary_q25', 'salary_q50', 'salary_q75', 'frequency', 
                    'popular_profession_id', 'popular_profession_name', <professions>]
        """
        filename = os.path.join(self.features_folder, 'skills.csv')
        self.skill_df.to_csv(filename, index=False, encoding='utf-8')


    def update_skill_df(self) -> None:
        """Update skill dataframe. Add weighed most popular profession to each skill"""

        log.info('Updating skill dataframe...')

        prof_frequency = self.prof_df.sort_values(by='prof_id').frequency.to_numpy()
        self.skill_df['popular_profession_id'] = (self.matrix / prof_frequency).argmax(axis=1)
        self.skill_df['popular_profession_name'] = \
            self.skill_df.popular_profession_id.apply(lambda x: self.prof_index_to_prof_name[x])

        matrix_colsum = self.matrix.sum(axis=0)
        for k, v in self.prof_index_to_prof_name.items():
            self.skill_df[v] = self.skill_df['skill_id'].apply(lambda x: self.matrix[x, k] / matrix_colsum[k])

        log.info('Updated skill dataframe...')


    def skills_processing(self) -> None:
        """
        Make skill processing
        Save files:
            - 'skills.txt': all skills after corrections ordered by name
            - 'skill_original_to_index.pkl': dictionary skill name to index
            - 'skill_index_to_corrected.pkl': dictionary skill index to corrected skill name
        """

        skills = self._extract_skills()
        self.skill_original_to_index, self.skill_index_to_corrected = self._skills_corrections(skills)

        filename = os.path.join(self.features_folder, 'skills.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines([x + '\n' for x in sorted(self.skill_index_to_corrected.values())])

        filename = os.path.join(self.features_folder, 'skill_original_to_index.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.skill_original_to_index, f)

        filename = os.path.join(self.features_folder, 'skill_index_to_corrected.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.skill_index_to_corrected, f)

        log.info('All skills count = %s, after correction = %s',
                len(self.skill_original_to_index), len(self.skill_index_to_corrected))

        self.skill_df = self._create_skill_df(self.skill_original_to_index, self.skill_index_to_corrected)


    def save_prof_df(self) -> None:
        """Save skill dataframe to file
        'prof.csv': dataframe with columns:
                ['prof_name', 'prof_id', 'salary_q25', 'salary_q50', 'salary_q75', 'frequency', 'popular_skills']
        """

        filename = os.path.join(self.features_folder, 'prof.csv')
        self.prof_df.to_csv(filename, index=False, encoding='utf-8')


    def update_prof_df(self, top_n: int = 10) -> None:
        """
        Update prof_df
        Add weighed most popular profession to each skill"""

        log.info('Updating prof data frame...')
        self.prof_df['popular_skills'] = self.prof_df['prof_id'].apply(
            lambda i: ', '.join([self.skill_index_to_corrected[x] for x in self.matrix[:,i].argsort()[::-1]][:10]))
        log.info('Updated prof data frame...')

    def professions_processing(self) -> None:
        """Professions preprocess:
            Add column 'prof_set' to self.df

        Save files:
            - 'quety_to_prof_index.pkl': dict query name to professio index
            - 'prof_index_to_prof_name': dict profession index to profession name

        """

        log.info('Processing professions...')

        # map query to profession
        prof_map = dict(zip(self.professions.query_name, self.professions.profession))


        # calculate 'prof_set' column
        r = re.compile('\\\\|////|,')
        def query_to_prof_set(s):
            query_set = {x.strip(" '") for x in re.split(r, s.strip('[]'))} - {''}
            prof_set = set([])
            for q in query_set:
                prof_set.add(prof_map.get(q, q))
            return prof_set

        self.df['prof_set'] = self.df['query'].apply(query_to_prof_set)

        self.quety_to_prof_index = {}
        self.prof_index_to_prof_name = {}
        self.prof_to_index = {}

        # get unique queries
        query_set_series = self.df['query'].apply(
            lambda s : {x.strip(" '") for x in re.split(r, s.strip('[]'))} - {''})
        all_queries = set().union(*query_set_series.to_list())

        # calcutate quety_to_prof_index and prof_index_to_prof_name
        new_index = 0
        for q in all_queries:
            prof = prof_map.get(q, q)
            index = self.prof_to_index.get(prof, new_index)
            self.prof_to_index[prof] = index
            self.prof_index_to_prof_name[index] = prof
            self.quety_to_prof_index[q] = index
            if new_index == index:
                new_index += 1

        # save quety_to_prof_index and prof_index_to_prof_name
        filename = os.path.join(self.features_folder, 'quety_to_prof_index.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.quety_to_prof_index, f)

        filename = os.path.join(self.features_folder, 'prof_index_to_prof_name.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.prof_index_to_prof_name, f)

        # create prof_df data frame
        self.prof_df = pd.DataFrame()
        self.prof_df['prof_name'] = self.prof_index_to_prof_name.values()
        self.prof_df['prof_id'] = self.prof_index_to_prof_name.keys()
        
        salary_q25 = {}
        salary_q50 = {}
        salary_q75 = {}
        prof_freq = {}

        for p in self.prof_df.prof_name:
            p_df = self.df[self.df['prof_set'].apply(lambda x: p in x)]            
            prof_freq[p] = p_df.shape[0] / self.df.shape[0]
            p_df = p_df[p_df.salary]
            if p_df.shape[0] > 0:
                salaries = (p_df['salary_from'] + p_df['salary_to']) / 2
                salary_q25[p] = round(salaries.quantile(0.25))
                salary_q50[p] = round(salaries.quantile(0.50))
                salary_q75[p] = round(salaries.quantile(0.75))

        self.prof_df['salary_q25'] = self.prof_df.prof_name.apply(lambda x: salary_q25.get(x, None))
        self.prof_df['salary_q50'] = self.prof_df.prof_name.apply(lambda x: salary_q50.get(x, None))
        self.prof_df['salary_q75'] = self.prof_df.prof_name.apply(lambda x: salary_q75.get(x, None))
        self.prof_df['frequency'] = self.prof_df.prof_name.apply(lambda x: prof_freq.get(x, None))

        log.info('Processed professions')

    def rel_matrix_processing(self) -> np.array:
        """
        Return and save skill-profession relationship matrix
        Matrix dim is about 8000x10 therefore numpy is enough

        Save matrix to 'matrix.csv' file
        """

        log.info('Creating relationship matrix...')
        
        self.matrix = np.zeros((len(self.skill_index_to_corrected), len(self.prof_index_to_prof_name)))
        for row in self.df[['prof_set', 'skill_set']].to_numpy():
            for prof in row[0]:
                prof_id = self.prof_to_index[prof]
                for skill in row[1]:
                    skill_id = self.skill_original_to_index[skill]
                    self.matrix[skill_id, prof_id] += 1

        filename = os.path.join(self.features_folder, 'matrix.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.matrix, f)

        log.info('Created relationship matrix...')

        return self.matrix

    def process(self) -> None:
        """Conduct all process. Input and output date in files"""

        self.skills_processing()
        self.professions_processing()
        self.rel_matrix_processing()
        self.update_skill_df()
        self.save_skill_df()
        self.update_prof_df()
        self.save_prof_df()
        log.info('Feature procissing completed')


if __name__ == '__main__':

    FeaturesProcessor().process()

    
