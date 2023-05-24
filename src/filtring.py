"""Filtring relevant vacancies to DataScience"""
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from pymystem3 import Mystem
import nltk
from nltk.corpus import stopwords
import lightgbm as lgb
from src.utils.logger import configurate_logger

log = configurate_logger('Preprocessor')



class RelevantVacancyClassifier():
    """Classifacator to filter relavant vacancies to Data Science
    imput: dataframe with column 'name'
    output: relevant or not (0/1)
    """

    _RANDOM_SEED = 42

    def __init__(self):

        nltk.download('stopwords')
        self._stop_words = set(stopwords.words('english')).union(stopwords.words('russian'))

        self._m = Mystem()

    def _lemmatize(self, X):
        def simplify(s):
            return s.lower(). \
                replace('"', ' '). \
                replace(",", ' '). \
                replace('(', ' '). \
                replace(')', ' '). \
                replace('\\', ' '). \
                replace('-', ' '). \
                replace('/', ' '). \
                replace('.', ' ')
        
        X = [[x.strip() for x in self._m.lemmatize(simplify(r)) if x.strip() != ''] for r in X.name]
        return [' '.join(x) for x in X]

    def fit(self, X, y):

        X = self._lemmatize(X)

        self._vectorizer = CountVectorizer(stop_words=list(self._stop_words))
        #self._vectorizer = TfidfVectorizer(stop_words=list(self._stop_words))

        #self._model = LogisticRegression(class_weight='balanced')
        self._model = lgb.LGBMClassifier(n_estimators=500, random_state=self._RANDOM_SEED)

        X = self._vectorizer.fit_transform(X)
        self._model.fit(X.astype(float), y)

    def predict(self, X):
        
        log.info('Filtring relevatnt vacancies... Total rows: %s', X.shape[0])

        X_lemm = self._lemmatize(X)
        XX = self._vectorizer.transform(X_lemm)
        y = self._model.predict(XX.astype(float))

        # коррекция на базе правил
        key_words = ['NLP', 'MLOps', 'DataOps', 'Computer Vision']
        for i, x in enumerate(X_lemm):
            if any([(w.lower() in x.split(' ')) for w in key_words]):
                y[i] = 1

        log.info('Filtred relevatnt vacancies. Result rows: %s', sum(y))

        return y
    
    def dump(self, filename: str):
        dumped = [self._vectorizer, self._model]
        with open(filename, 'wb') as f:
            pickle.dump(dumped, f)

    def load(self, filename: str):
        with open(filename, 'rb') as f:
            dumped = pickle.load(f)
            self._vectorizer = dumped[0]
            self._model = dumped[1]


if __name__ == '__main__':
    clf = RelevantVacancyClassifier()
    clf.load('models/RelevantVacancyClassifier.pkl')

    df = pd.read_csv('data/processed/vacancies.csv', encoding='utf-8')

    df['predict'] = clf.predict(df[['name']])
    df = df[df.predict == 1].drop(columns=['predict'])

    df.to_csv('data/processed/vacancies.csv', encoding='utf-8', index=False)