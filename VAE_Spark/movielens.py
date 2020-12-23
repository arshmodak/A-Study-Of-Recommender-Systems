import pandas as pd
import numpy as np
import re
import os.path

class Movielens:
    def __init__(self):
        if os.path.isfile('data/movie_data_merge'):
            self.movie_data_merge = pd.read_pickle('data/movie_data_merge')
        else:
            self.ratings = pd.read_csv('data/ratings_small.csv')
            self.credits = pd.read_csv('data/credits.csv')
            self.keywords = pd.read_csv('data/keywords.csv')
            self.links = pd.read_csv('data/links.csv')
            self.movie_metadata = pd.read_csv('data/movies_metadata.csv')

            self.columns_to_choose = ['adult', 'budget', 'genres', 'id', 'imdb_id', 'original_language', 'original_title',
                             'overview', 'popularity', 'production_companies', 'production_countries', 'release_date',
                             'revenue', 'runtime', 'spoken_languages', 'status', 'title', 'vote_average', 'vote_count']

            self.numeric_columns = ['budget', 'id', 'imdb_id', 'popularity', 'revenue', 'runtime', 'vote_average',
                               'vote_count', 'score']
            self.categorical_columns = ['adult', 'genres', 'original_language', 'original_title', 'overview',
                                   'production_companies', 'production_countries', 'spoken_languages', 'status',
                                   'title']

            self.movie_metadata_changed = self.movie_metadata[self.columns_to_choose]
            self.preprocess_genres()
            self.preprocess_id()
            self.preprocess_production_company()
            self.preprocess_production_country()
            self.preprocess_spoken_language()
            self.addfeature_score()

    def preprocess_genres(self):
        def change_genres(row):
            rows = row[1:-1]
            matches = re.findall(r"\'name\'..\'\w+\'", rows)
            genre = ",".join([i.split(':')[1].replace("'", "") for i in matches])
            return genre
        self.movie_metadata_changed['genres'] = self.movie_metadata_changed['genres'].apply(change_genres)

    def preprocess_id(self):
        def change_typeid(row):
            try:
                return int(row)
            except ValueError:
                return None
        self.movie_metadata_changed['id'] = self.movie_metadata_changed['id'].apply(change_typeid)
        self.movie_data_merge = self.movie_metadata_changed.merge(self.credits, on='id')

    def preprocess_production_company(self):
        def change_pc(row):
            import re
            if row is not np.NaN:
                rows = row[1:-1]
            else:
                rows = ''
            matches = re.findall(r"\'name\'..\'\w*.?\w*\s?\w*\s?\w*\s?\w*\s?\w*\s?\w*", rows)
            pc = ",".join([i.split(':')[1].replace("'", "") for i in matches])
            return pc
        self.movie_data_merge['production_companies'] = self.movie_data_merge['production_companies'].apply(change_pc)

    def preprocess_production_country(self):
        def change_pcon(row):
            if row is not np.NaN:
                rows = row[1:-1]
            else:
                rows = ''
            matches = re.findall(r"\'name\'..\'\w*.?\w*\s?\w*\s?\w*\s?\w*\s?\w*\s?\w*", rows)
            pc = ",".join([i.split(':')[1].replace("'", "") for i in matches])
            return pc
        self.movie_data_merge['production_countries'] = self.movie_data_merge['production_countries'].apply(change_pcon)

    def preprocess_spoken_language(self):
        def change_spoken_lang(row):
            if row is not np.NaN:
                rows = row[1:-1]
            else:
                rows = ''
            matches = re.findall(r"\'name\'..\'\w*.?\w*\s?\w*\s?\w*\s?\w*\s?\w*\s?\w*", rows)
            pc = ",".join([i.split(':')[1].replace("'", "") for i in matches])
            return pc
        self.movie_data_merge['spoken_languages'] = self.movie_data_merge['spoken_languages'].apply(change_spoken_lang)

    def addfeature_score(self):
        m = self.movie_data_merge.vote_count.quantile(0.9)
        C = self.movie_data_merge.vote_average.mean()
        R = self.movie_data_merge['vote_average']
        v = self.movie_data_merge['vote_count']
        self.movie_data_merge['score'] = (v / (v + m) * R) + (m / (v + m) * C)
        self.movie_data_merge = self.movie_data_merge.sort_values(by='score', ascending=False)
        self.movie_data_merge.to_pickle('data/movie_data_merge')