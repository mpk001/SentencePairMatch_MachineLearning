# coding:utf-8
import numpy as np
import csv
import datetime
from sklearn.ensemble import GradientBoostingClassifier
import os
import pandas as pd
from sklearn import metrics
cwd = os.getcwd()

def load_data(datapath):
        data_train = pd.read_csv(datapath, sep='\t', encoding='utf-8')
        print data_train.shape

        qid1 = []
        qid2 = []
        labels = []
        count = 0
        for idx in range(data_train.id.shape[0]):
        # for idx in range(400):
        #     count += 1
        #     if count == 21: break
            print idx
            q1 = data_train.qid1[idx]
            q2 = data_train.qid2[idx]

            qid1.append(q1)
            qid2.append(q2)
            labels.append(data_train.is_duplicate[idx])

        return qid1, qid2, labels

def load_doc2vec(word2vecpath):
        f = open(word2vecpath)
        embeddings_index = {}
        count = 0
        for line in f:
            # count += 1
            # if count == 10000: break
            values = line.split('\t')
            id = values[0]
            print id
            coefs = np.asarray(values[1].split(), dtype='float32')
            embeddings_index[int(id)+1] = coefs
        f.close()
        print('Total %s word vectors.' % len(embeddings_index))

        return embeddings_index

def sentence_represention(qid, embeddings_index):
        vectors = np.zeros((len(qid), 100))
        for i in range(len(qid)):
            print i
            vectors[i] = embeddings_index.get(qid[i])

        return vectors

def main():
    start = datetime.datetime.now()
    datapath = 'D:/dataset/quora/quora_duplicate_questions_Chinese_seg.tsv'
    doc2vecpath = "D:/dataset/quora/vector2/quora_duplicate_question_doc2vec_100.vector"
    qid1, qid2, labels = load_data(datapath)
    embeddings_index = load_doc2vec(word2vecpath=doc2vecpath)
    vectors1 = sentence_represention(qid1, embeddings_index)
    vectors2 = sentence_represention(qid2, embeddings_index)
    vectors = np.hstack((vectors1, vectors2))
    labels = np.array(labels)
    VALIDATION_SPLIT = 10000
    VALIDATION_SPLIT0 = 1000
    indices = np.arange(vectors.shape[0])
    np.random.shuffle(indices)
    vectors = vectors[indices]
    labels = labels[indices]
    train_vectors = vectors[:-VALIDATION_SPLIT]
    train_labels = labels[:-VALIDATION_SPLIT]
    test_vectors = vectors[-VALIDATION_SPLIT:]
    test_labels = labels[-VALIDATION_SPLIT:]
    # train_vectors = vectors[:VALIDATION_SPLIT0]
    # train_labels = labels[:VALIDATION_SPLIT0]
    # test_vectors = vectors[-VALIDATION_SPLIT0:]
    # test_labels = labels[-VALIDATION_SPLIT0:]

    gbdt = GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=3, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
    print '***********************training************************'
    gbdt.fit(train_vectors, train_labels)

    print '***********************predict*************************'
    prediction = gbdt.predict(test_vectors)
    accuracy = metrics.accuracy_score(test_labels, prediction)
    acc = gbdt.score(test_vectors, test_labels)
    print accuracy
    print acc

    end = datetime.datetime.now()
    print end-start


if __name__ == '__main__':
    main() # the whole one model