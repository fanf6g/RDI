import logging

import numpy as np
import pandas as pd
import pymongo
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from model import Model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


class MIBOS(Model):
    def __init__(self, db, cv, attr, queryfilter):
        super().__init__(db, cv)
        self.attr = attr
        self.queryfilter = queryfilter

    def _extract(self, coll, attr):
        cur = coll.find()
        res = [rec[attr] for rec in cur]
        return res

    def mibos(self, K=8):

        def _compatible(s1, s2):
            return 1 if len(s1) == 0 or len(s2) == 0 or s1 == s2 else 0

        def _match(s1, s2):
            return 1 if len(s1) > 0 and s1 == s2  else 0

        v_match = np.vectorize(_match)
        v_compatible = np.vectorize(_compatible)

        def equal(train_attrs, test_attrs):
            # attrs = []
            # attrs.extend(train_attrs)
            # attrs.extend(test_attrs)

            # self.cv.fit(attrs)

            # train_features = self.cv.transform(train_attrs)
            # test_features = self.cv.transform(test_attrs)

            row_ind = []
            col_ind = []

            train_sets = [str(row).lower() for row in train_attrs]
            test_sets = [str(row).lower() for row in test_attrs]

            for (i, s_i) in enumerate(test_sets):
                # si = set(fi.indices)
                j_ind = v_match(s_i, train_sets)
                j_ind[i] = 0
                ind = np.where(j_ind == 1)
                j_len = len(ind[0])
                if (j_len > 0):
                    row_ind.extend([i] * j_len)
                    col_ind.extend(ind[0])
                    # print(i, train_attrs[i], j_len)

            pk = csr_matrix((np.ones(len(row_ind), dtype='int8'), (row_ind, col_ind)),
                            shape=(len(test_attrs), len(train_attrs)))

            return pk

        def compatible(train_attrs, test_attrs):
            # attrs = []
            # attrs.extend(train_attrs)
            # attrs.extend(test_attrs)
            #
            # self.cv.fit(attrs)
            #
            # train_features = self.cv.transform(train_attrs)
            # test_features = self.cv.transform(test_attrs)

            row_ind = []
            col_ind = []

            train_sets = [str(row).lower() for row in train_attrs]
            test_sets = [str(row).lower() for row in test_attrs]

            for (i, s_i) in enumerate(test_sets):
                # si = set(fi.indices)
                j_ind = v_compatible(s_i, train_sets)
                j_ind[i] = 0
                ind = np.where(j_ind == 1)
                j_len = len(ind[0])
                if (j_len > 0):
                    row_ind.extend([i] * j_len)
                    col_ind.extend(ind[0])
                    # print(i, train_attrs[i], j_len)

            pk = csr_matrix((np.ones(len(row_ind), dtype='int8'), (row_ind, col_ind)),
                            shape=(len(test_attrs), len(train_attrs)))

            return pk

        train = self.db.train
        valid = self.db.valid
        test = self.db.test

        df_train = pd.DataFrame(list(train.find()))[[CITY, POSTAL_CODE, LBL]]
        df_valid = pd.DataFrame(list(valid.find()))[[CITY, POSTAL_CODE, LBL]]
        df_test = pd.DataFrame(list(test.find()))[[CITY, POSTAL_CODE, LBL]]

        train_city = df_train.pop(CITY).values.tolist()
        train_city = [str(s) for s in train_city]
        # train_state = df_train.pop(STATE).values.tolist()
        # train_state = [str(s) for s in train_state]
        train_postalcode = df_train.pop(POSTAL_CODE).values.tolist()
        train_postalcode = [str(s) for s in train_postalcode]
        train_label = df_train.pop(LBL).values.tolist()

        valid_city = df_valid.pop(CITY).values.tolist()
        valid_city = [str(s) for s in valid_city]
        # valid_state = df_valid.pop(STATE).values.tolist()
        # valid_state = [str(s) for s in valid_state]
        valid_postalcode = df_valid.pop(POSTAL_CODE).values.tolist()
        valid_postalcode = [str(s) for s in valid_postalcode]
        valid_label = df_valid.pop(LBL).values.tolist()

        test_city = df_test.pop(CITY).values.tolist()
        test_city = [str(s) for s in test_city]
        # test_state = df_test.pop(STATE).values.tolist()
        # test_state = [str(s) for s in test_state]
        test_postalcode = df_test.pop(POSTAL_CODE).values.tolist()
        test_postalcode = [str(s) for s in test_postalcode]
        test_label = df_test.pop(LBL).values.tolist()

        cities = []
        cities.extend(train_city)
        cities.extend(valid_city)
        # cities.extend(test_city)



        # states = []
        # states.extend(train_state)
        # states.extend(valid_state)
        # authors.extend(test_author)

        posts = []
        posts.extend(train_postalcode)
        posts.extend(valid_postalcode)

        lbl = []
        lbl.extend(train_label)
        lbl.extend(valid_label)
        # lbl.extend(test_label)
        a_lbl = np.array(lbl)
        a_test_lbl = np.array(test_label)

        # print(len(posts))
        # print(len(test_city))

        logging.info('C_city')
        C_city = compatible(cities, test_city)
        # logging.info('C_states')
        # C_states = compatible(states, test_state)
        logging.info('C_posts')
        C_posts = compatible(posts, test_postalcode)

        logging.info('E_city')
        E_city = equal(cities, test_city)
        # logging.info('E_states')
        # E_states = equal(states, test_state)
        logging.info('E_posts')
        E_posts = equal(posts, test_postalcode)

        logging.info('P_mul')
        P_mul = C_city.multiply(C_posts)
        logging.info('P_add')
        P_add = E_city + E_posts
        P_ = P_mul.multiply(P_add)
        P_.eliminate_zeros()

        print(P_.shape)
        print(type(P_))
        print(P_)

        n_match = 0
        sn = 0
        for (i, ns_i) in enumerate(P_):
            ns = ns_i.indices
            if (ns.size > 0):
                sim = ns_i.data
                max_occur = np.where(sim == sim.max())[0]
                ind = ns[max_occur]
                # sn = sn + 1
                can = np.unique(a_lbl[ind])
                print(i, a_test_lbl[i], can)
                if can.size == 1:
                    sn = sn + 1
                    if a_test_lbl[i] == can[0]:
                        n_match = n_match + 1
            pass

        print(n_match, sn)


if __name__ == "__main__":
    _ID = '_id'
    LBL = 'lbl'
    HOTELID = 'hotelID'
    COUNTRY = 'country'
    AIRPORTCODE = 'airport code'
    POSTAL_CODE = 'postal code'
    HOTEL = 'hotel'
    ADDRESS = 'address'
    CITY = 'city'
    STATE = 'state'
    COUNTRY = 'country'
    LBL = 'lbl'
    client = pymongo.MongoClient('localhost', 27017)
    queryfilter = {ADDRESS: {'$exists': 'true'},
                   CITY: {'$exists': 'true'},
                   # AUHTOR: {'$exists': 'true'},
                   COUNTRY: {'$exists': 'true'},
                   }
    client = pymongo.MongoClient('localhost', 27017)

    db1 = client['hotel']

    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    knn = MIBOS(db1, cv12, AIRPORTCODE, queryfilter)

    knn.mibos(100)

    MAS = []

    pass
