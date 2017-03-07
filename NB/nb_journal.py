import logging

import numpy
import pandas as pd
import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

from model import Model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')

client = pymongo.MongoClient('localhost', 27017)
# db = client['accuracy']
# db = client['movies']

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'


class NB(Model):
    def __init__(self, db, cv=CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1, 1), dtype='int16'), alpha=1.0e-2):
        super().__init__(db, cv)
        self.clf = BernoulliNB(alpha=alpha)
        self.alpha = alpha

    def training(self, features, labels):
        A_feature = features.toarray()

        n_records, n_features = A_feature.shape
        n_class = max(labels) + 1
        # self.proba = numpy.zeros((n_features, n_class), dtype='float32')
        self.feature_cls_counts = numpy.zeros((n_class, n_features), dtype='int16')
        self.feature_counts = numpy.zeros((n_features), dtype='int16')

        for (row, cls) in zip(A_feature, labels):
            self.feature_cls_counts[cls] += row

        self.feature_counts = numpy.sum(self.feature_cls_counts, axis=0)

        smoothed_fc = self.feature_cls_counts + self.alpha
        smoothed_cc = self.feature_counts + self.alpha * n_class

        self.proba = smoothed_fc / smoothed_cc
        self.idf = numpy.log(n_records / (1.0 + smoothed_cc))

        m_f = self.proba * self.proba
        self.f_entropy = numpy.sum(m_f, axis=0)
        self.clf.fit(features, labels)

    def inference(self, features, labels):
        # col_index = numpy.array(self.f_entropy >= self.f_theta, dtype='int16')
        # print('feature selection: {0}'.format(numpy.unique(col_index, return_counts=True)))

        A_feature = features.toarray()
        row_index = numpy.array(numpy.where(A_feature.sum(axis=1) > 0)[0])
        predictions = self.clf.predict(features)
        # return numpy.unique(predictions == labels, return_counts=True)

        # res = numpy.unique(numpy.array(predictions)[row_index] == numpy.array(labels)[row_index], return_counts=True)
        res = numpy.unique(numpy.array(predictions) == numpy.array(labels), return_counts=True)
        return res


# if __name__ == "__main__":
#     # attr = 'producer'
#
#     model = NB(db)
#
#     titles_train = model.extract('train', 'title')
#     actors_train = model.extract('train', 'actors')
#     director_train = model.extract('train', 'director')
#     at_train = [a + ' ' + d + ' ' + t for (a, d, t) in zip(actors_train, director_train, titles_train)]
#
#     # author_train = model.extract('train', 'author')
#     # at_train = [a + ' ' + t for (a, t) in zip(author_train, titles_train)]
#
#     titles_test = model.extract('test', 'title')
#     actors_test = model.extract('test', 'actors')
#     director_test = model.extract('test', 'director')
#     at_test = [a + ' ' + d + ' ' + t for (a, d, t) in zip(actors_test, director_test, titles_test)]
#
#     # author_test = model.extract('test', 'author')
#     # at_test = [a + ' ' + t for (a, t) in zip(author_test, titles_test)]
#
#     # X_train, X_test = model.featuring(titles_train, titles_test)
#     X_train, X_test = model.featuring(at_train, at_test)
#     y_train = model.extract('train', 'lbl')
#     y_test = model.extract('test', 'lbl')
#
#     model.training(X_train, y_train)
#     model.inference(X_test, y_test)
#
#     # clf = BernoulliNB()
#     # clf.fit(X_train, y_train)
#     # y_predict = clf.predict(X_test)
#     # print(numpy.unique(y_test == y_predict, return_counts=True))
#     pass

def dblp(cv=CountVectorizer(dtype='int16'), alpha=1.0e-8):
    db = client['accuracy']
    model = NB(db, cv=cv, alpha=alpha)
    train = db.train
    valid = db.valid
    test = db.test

    pd_train = pd.DataFrame(list(train.find()))[[AUTHOR, TITLE, LBL]]
    pd_valid = pd.DataFrame(list(valid.find()))[[AUTHOR, TITLE, LBL]]
    pd_test = pd.DataFrame(list(test.find()))[[AUTHOR, TITLE, LBL]]

    y_train = pd_train.pop(LBL).values.tolist()
    y_valid = pd_valid.pop(LBL).values.tolist()
    y_test = pd_test.pop(LBL).values.tolist()
    # y_train.extend(y_valid)

    print(pd_train.head())

    at_train = [' '.join(data.map(lambda x: str(x))) for index, data in pd_train.iterrows()]
    logging.info('load dataframe pd_train!')
    at_valid = [' '.join(data.map(lambda x: str(x))) for index, data in pd_valid.iterrows()]
    logging.info('load dataframe pd_valid')
    at_test = [' '.join(data.map(lambda x: str(x))) for index, data in pd_test.iterrows()]
    logging.info('load dataframe pd_test!')

    # at_train.extend(at_valid)

    # at_train = [a + ' ' + t for (a, t) in zip(train_author, train_title)]
    # at_valid = [a + ' ' + t for (a, t) in zip(valid_author, valid_title)]
    # at_test = [a + ' ' + t for (a, t) in zip(test_author, test_title)]

    X_train, X_valid, X_test = model.featuring(at_train, at_valid, at_test)

    model.training(X_train, y_train)
    return model.inference(X_test, y_test)


if __name__ == "__main__":
    ID = '_id'
    AUTHOR = 'author'
    TITLE = 'title'
    JOURNAL = 'journal'
    YEAR = 'year'
    LBL = 'lbl'
    # dblp()
    # cv11 = CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1, 1), dtype='int16', stop_words='english')
    cv12 = CountVectorizer(ngram_range=(1, 1), dtype='int16', stop_words='english')
    # cv21 = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 1), dtype='int16', stop_words='english')
    # cv22 = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), dtype='int16', stop_words='english')

    print(dblp(cv12, alpha=1e-8))
    print(str(cv12))
