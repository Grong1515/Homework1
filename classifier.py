from random import shuffle
import numpy as np

import xml.etree.ElementTree as ET
import word2vec
from sklearn.feature_selection import RFECV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


BANK_TRAIN_BASE = 'data/bank_train_2016.xml'
BANK_TEST_BASE = 'data/banks_test_etalon.xml'
TKK_TRAIN_BASE = 'data/tkk_train_2016.xml'
TKK_TEST_BASE = 'data/tkk_test_etalon.xml'

BANK_LIST = ['rshb', 'alfabank', 'vtb', 'bankmoskvy', 'raiffeisen', 'sberbank', 'uralsib', 'gazprom']
TKK_LIST = ['beeline', 'mts', 'megafon', 'tele2', 'rostelecom', 'komstar', 'skylink']


def load_data(file, names):
    tree = ET.parse(file)
    database = tree.getroot().find('database')

    train_set = []
    answer_set = []

    for table in database.iter('table'):
        train_set.append(table.find('./*[@name="text"]').text)
        answer = 0
        for item in names:
            if answer == 0:
                value = table.find('./*[@name="{}"]'.format(item)).text
                if value == '-1':
                    answer = 'negative'
                    break
                elif value == '0':
                    answer = 'neutral'
                    break
                elif value == '1':
                    answer = 'positive'
                    break
        answer_set.append(answer)

    mix = list(range(len(train_set)))
    shuffle(mix)

    return np.array(shuffle_set(train_set, mix)), np.array(shuffle_set(answer_set, mix))


def shuffle_set(set, mix):
    response = list(range(len(mix)))
    for i in list(range(len(mix))):
        response[mix[i]] = set[i]
    return response


def train_and_test(train_base, test_base, items_list):
    X, y = load_data(train_base, items_list)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    model = MLPClassifier(alpha=1e-5, solver="lbfgs",
                          hidden_layer_sizes=(100,), random_state=1)
    # model = LogisticRegression()
    kf = KFold(n_splits=10)
    print('CROSS VALIDATION')
    i = 1
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        prediction = model.predict(X[test])
        print('Number ot iteration: ' + str(i))
        print(metrics.classification_report(y[test], prediction))
        i += 1

    # print(cross_val_score(model, X, y))
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print('-'*60)
    print('Results of training:')
    print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))

    X_test, y_test = load_data(test_base, items_list)
    X_test = vectorizer.transform(X_test)
    predicted = model.predict(X_test)
    print('Results of testing:')
    print(metrics.classification_report(y_test, predicted))
    # print(metrics.confusion_matrix(y_test, predicted))


if __name__ == '__main__':
    # uncomment the line you need
    train_and_test(BANK_TRAIN_BASE, BANK_TEST_BASE, BANK_LIST)
    # train_and_test(TKK_TRAIN_BASE, TKK_TEST_BASE, TKK_LIST)