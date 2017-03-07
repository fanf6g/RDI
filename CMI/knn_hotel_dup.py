import logging
import random

import numpy as np
import pandas as pd
import pymongo
from sklearn.feature_extraction.text import CountVectorizer

from model import Model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')

# client = pymongo.MongoClient('localhost', 27017)
# queryfilter = {TITLE: {'$exists': 'true'},
#                JOURNAL: {'$exists': 'true'},
#                # AUHTOR: {'$exists': 'true'},
#                YEAR: {'$exists': 'true'},
#                }

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'


class KNN(Model):
    def __init__(self, db, cv, attr, queryfilter):
        super().__init__(db, cv)
        self.attr = attr
        self.queryfilter = queryfilter

    def clean(self):
        '''
        清理数据表
        :return:
        '''
        names = ['test', 'train', 'valid', 'sampledb', 'dblptmp']
        for name in names:
            logging.info('delete %s' % (name))
            self.db.get_collection(name).drop()

    def sampling(self, n_samples=50000, dup_ratio=0.5):
        dblp = self.db.hoteldata
        sampledb = self.db.sampledb

        cur = dblp.find(self.queryfilter).limit(n_samples)

        tuples = []
        for rec in cur:
            rec.pop('_id')
            tuples.append(rec)

        np.random.seed(17)
        n_sample = int(n_samples * dup_ratio)
        n_dup = [len(d) for d in np.array_split(range(n_sample), 5)]

        dup = []
        random.seed(17)
        for n in n_dup:
            dup.extend(random.sample(tuples, n))

        dup.extend(tuples)
        # random.seed(17)
        # random.shuffle(dup)

        for rec in dup:
            sampledb.save(rec.copy())

    def shuffle_label(self, seed=17):
        dblp = self.db.sampledb
        tmp = self.db.dblptmp

        journals = dblp.distinct(self.attr)
        journals.sort()

        jkv = enumerate(journals)

        records = []

        m = {}
        # y = {}
        for (i, v) in jkv:
            m[v] = i

        cur = dblp.find(self.queryfilter)
        for rec in cur:
            del rec['_id']
            rec['lbl'] = m[rec[self.attr]]
            records.append(rec)

        random.seed(seed)
        random.shuffle(records)
        for rec in records:
            tmp.insert(rec)

    def split(self, train_count=40000, valid_count=5000, test_count=5000):
        dblptmp = self.db.dblptmp
        train = self.db.train
        valid = self.db.valid
        test = self.db.test

        docs = []
        cur = dblptmp.find()
        for rec in cur:
            docs.append(rec)

        train_slice = slice(0, train_count)
        valid_slice = slice(train_count, train_count + valid_count)
        test_slice = slice(train_count + valid_count, train_count + valid_count + test_count)

        train_docs = docs[train_slice]
        valid_docs = docs[valid_slice]
        test_docs = docs[test_slice]

        for rec in train_docs:
            train.save(rec)

        for rec in valid_docs:
            valid.save(rec)

        for rec in test_docs:
            test.save(rec)

    def _extract(self, coll, attr):
        cur = coll.find()
        res = [rec[attr] for rec in cur]
        return res

    def knn_pred(self, K=200):

        db = client['hotel']

        col_train = db.get_collection(TRAIN)
        col_valid = db.get_collection(VALID)
        col_test = db.get_collection(TEST)

        cur_train = col_train.find()
        cur_valid = col_valid.find()
        cur_test = col_test.find()

        pd_train = pd.DataFrame(list(cur_train))[[ADDRESS, CITY, COUNTRY, CURRENCY, HOTEL, POSTAL_CODE, STATE, LBL]]
        pd_valid = pd.DataFrame(list(cur_valid))[[ADDRESS, CITY, COUNTRY, CURRENCY, HOTEL, POSTAL_CODE, STATE, LBL]]
        pd_test = pd.DataFrame(list(cur_test))[[ADDRESS, CITY, COUNTRY, CURRENCY, HOTEL, POSTAL_CODE, STATE, LBL]]

        print(pd_train.head())

        train_label = pd_train.pop(LBL).values.tolist()
        valid_label = pd_valid.pop(LBL).values.tolist()
        test_label = pd_test.pop(LBL).values.tolist()
        train_label.extend(valid_label)

        at_train = [' '.join(data.map(lambda x: str(x))) for index, data in pd_train.iterrows()]
        logging.info('load dataframe pd_train!')
        at_valid = [' '.join(data.map(lambda x: str(x))) for index, data in pd_valid.iterrows()]
        logging.info('load dataframe pd_valid')
        at_test = [' '.join(data.map(lambda x: str(x))) for index, data in pd_test.iterrows()]
        logging.info('load dataframe pd_test!')

        at_train.extend(at_valid)
        at = []
        at.extend(at_train)
        # at.extend(at_valid)
        at.extend(at_test)
        self.cv.fit(at)
        train_data = self.cv.transform(at_train)
        test_data = self.cv.transform(at_test)

        m_test = test_data.sum(axis=1)
        m_train = train_data.sum(axis=1)
        m_intersect = (test_data.dot(train_data.T)).toarray()
        m_union = np.array(m_test + m_train.transpose() - m_intersect + 1.0e-4)

        logging.info("computing top_k")
        top_k_idx = []
        count = 0
        '''cnt_matrix ./ cnt2_matrix = sim(test_data, train_data)'''
        for num, denorm in zip(m_intersect, m_union):
            count += 1
            sim = num / denorm
            di = np.argsort(-sim)[0:K]
            top_k_idx.append(di)
            if (count % 100 == 0):
                logging.info(str(count))

        logging.info("computing top_k")
        # top_k_idx = np.argsort(-M_sim, axis=1)[:, 0:K]

        self.top_k_idx = np.array(top_k_idx)
        self.train_label = np.array(train_label)
        self.test_label = np.array(test_label)

    def knn_verify(self):
        '''
        验证 CMI 方案的准确率.
        :return:
        '''
        knn_labels = self.train_label[self.top_k_idx]
        mlh_labels = []
        for knn_label in knn_labels:
            label_counts = np.unique(knn_label, return_counts=True)
            di = np.argmax(label_counts[1])
            mlh_label = label_counts[0][di]
            mlh_labels.append(mlh_label)

        knn_pred = np.array(mlh_labels)
        res = np.sum(knn_pred == self.test_label)
        print(res)
        return res


if __name__ == "__main__":
    _ID = '_id'
    HOTELID = 'hotelID'
    COUNTRY = 'country'
    CURRENCY = 'currency'
    POSTAL_CODE = 'postal code'
    AIRPORTCODE = 'airport code'
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
    db1 = client['hotel']

    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    knn = KNN(db1, cv12, AIRPORTCODE, queryfilter)
    knn.clean()
    dup_ratio = 0
    knn.sampling(n_samples=2000, dup_ratio=dup_ratio)
    knn.shuffle_label()
    rt = 5
    knn.split(train_count=int(800 // 5 * rt * (1 + dup_ratio)), valid_count=int(400 // 5 * rt * (1 + dup_ratio)),
              test_count=int(800 // 5 * rt * (1 + dup_ratio)))

    knn.knn_pred(1)
    knn.knn_verify()

    pass
