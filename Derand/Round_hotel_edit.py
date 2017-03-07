import logging
import pickle

import cvxpy
import jellyfish
import numpy as np
import pymongo
from sklearn.feature_extraction.text import CountVectorizer

from model import Model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')

client = pymongo.MongoClient('localhost', 27017)

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
CITY = 'city'
POSTCODE = 'postal code'
AIRPORT = 'airport code'
# TYPE = 'type'
LBL = 'lbl'


class Round(Model):
    def __init__(self, db, cv=CountVectorizer(min_df=2, ngram_range=(1, 2), dtype='int16')):
        super().__init__(db, cv)

    def training(self, features, labels):
        airport_train = self.extract(TRAIN, AIRPORT)
        city_train = self.extract(TRAIN, CITY)
        postcode_train = self.extract(TRAIN, POSTCODE)
        # types_train = self.extract(TRAIN, TYPE)

        airport_train.extend(self.extract(VALID, AIRPORT))
        city_train.extend(self.extract(VALID, CITY))
        postcode_train.extend(self.extract(VALID, POSTCODE))
        # types_train.extend(self.extract(VALID, TYPE))

        self.airport_train = list(map(lambda x: str(x), airport_train))
        self.city_train = list(map(lambda x: str(x), city_train))
        self.postcode_train = list(map(lambda x: str(x), postcode_train))
        # self.types_train = list(map(lambda x: str(x), types_train))
        pass

    def inference(self, features, labels):
        airport_test = self.extract(TEST, AIRPORT)
        city_test = self.extract(TEST, CITY)
        postcode_test = self.extract(TEST, POSTCODE)

        # types_test = self.extract(TEST, TYPE)

        # self.journals_test = list(map(lambda x: str(x), journals_test))
        # self.titles_test = list(map(lambda x: str(x), titles_test))
        # self.authors_test = list(map(lambda x: str(x), authors_test))
        # self.types_test = list(map(lambda x: str(x), types_test))

        def v_sim(tests, trains):
            recs = []
            recs.extend(trains)
            recs.extend(tests)
            cv = CountVectorizer(dtype='int16', stop_words='english')
            cv.fit(recs)
            logging.info(len(cv.vocabulary_))
            train_data = cv.transform(trains)
            test_data = cv.transform(tests)

            m_test = test_data.sum(axis=1)
            m_train = train_data.sum(axis=1)
            m_intersect = (test_data.dot(train_data.T)).toarray()
            m_union = np.array(m_test + m_train.transpose() - m_intersect + 1.0e-6)
            m_sim = m_intersect / m_union
            return np.array(m_sim, dtype='float32')

        # Nghb_title_test_train = self._edit_dist(self.titles_train, self.titles_test) <= 10
        # Nghb_author_test_train = self._edit_dist(self.authors_train, self.authors_test) <= 7
        # Nghb_type_test_train = self._edit_dist(self.types_train, self.types_test) <= 5

        # Nghb_title_test_test = self._edit_dist(self.titles_test, self.titles_test) <= 10
        # Nghb_author_test_test = self._edit_dist(self.titles_test, self.authors_test) <= 7
        # Nghb_type_test_test = self._edit_dist(self.names_test, self.types_test) <= 5

        self.airport_test = list(map(lambda x: str(x), airport_test))
        self.postcode_test = list(map(lambda x: str(x), postcode_test))
        self.city_test = list(map(lambda x: str(x), city_test))

        alpha_author = 1
        alpha_title = 1
        alpha_journal = 1

        Nghb_postcode_test_train = self._edit_dist(self.postcode_test, self.postcode_train) <= alpha_author
        Nghb_postcode_test_test = self._edit_dist(self.postcode_test, self.postcode_test) <= alpha_author
        Nghb_city_test_train = self._edit_dist(self.city_test, self.city_train) <= alpha_title
        Nghb_city_test_test = self._edit_dist(self.city_test, self.city_test) <= alpha_title

        Nghb_test_train = np.array(Nghb_city_test_train * Nghb_postcode_test_train)
        Nghb_test_test = Nghb_city_test_test * Nghb_postcode_test_test

        cities = np.array(sorted(set(self.airport_train)))
        n_lbl = len(cities)
        V = np.array(self._edit_dist(cities, cities) > alpha_journal, dtype='int8')
        for i in range(n_lbl):
            V[i, i] = 0

        city2id = dict([(v, k) for (k, v) in enumerate(cities)])

        n_test = len(airport_test)
        print(n_test)
        n_train = 720

        def compress(city2id, nghb, city_train):
            np_city = np.array(city_train)
            tmp = []
            for i, row in enumerate(nghb):
                nb = np_city[row]
                r = np.zeros(n_lbl, dtype='int8')
                for c in nb:
                    r[city2id[c]] = 1

                tmp.append(r)

            return np.array(tmp)

        W_test_train = compress(city2id, Nghb_test_train, self.airport_train)
        P_w_test_train = cvxpy.Parameter(n_test, n_lbl, name='candidates', sign='positive', value=W_test_train)

        P_w_test_test = cvxpy.Parameter(n_test, n_test, name='candidates', sign='positive', value=Nghb_test_test)
        print(Nghb_test_test.shape)

        print(np.sum(np.sum(W_test_train, axis=1) > 0))

        X_lp = cvxpy.Variable(n_test, n_lbl)
        objective = cvxpy.Maximize(cvxpy.sum_entries(cvxpy.mul_elemwise(P_w_test_train, X_lp)))

        '''约束: 公式(2)'''
        ONE = cvxpy.Parameter(n_lbl, sign='positive')
        ONE.value = np.ones(n_lbl, dtype='int16')

        X_lp * ONE
        P_v = cvxpy.Parameter(n_lbl, n_lbl, sign='positive', value=V)

        constraint = [X_lp * ONE <= 1,  # 约束(2)
                      P_w_test_test * X_lp * P_v <= 1,  # 约束(3)
                      X_lp >= 0, X_lp <= 1  # 约束(4)]
                      ]

        prob = cvxpy.Problem(objective, constraint)
        prob.solve()
        pred = np.argmax(X_lp.value, axis=1).flatten()
        print(pred)
        print(self.airport_test)
        gtrue = np.array([city2id.get(city, 0) for city in self.airport_test])
        print(gtrue)
        logging.info(np.sum(gtrue == pred))

        pass

    def _edit_dist(self, tests, train):
        V_edit = np.vectorize(jellyfish.levenshtein_distance)
        M_sim = np.array([V_edit(train, t) for t in tests])
        return M_sim


if __name__ == "__main__":
    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    db = client['hotel']
    model = Round(db, cv12)
    model.training([], [])
    model.inference([], [])

    pass
