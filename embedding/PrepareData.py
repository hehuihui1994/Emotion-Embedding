#!/usr/bin/env python
# encoding: utf-8


import re
import sys
import random
import numpy as np


def batch_index(length, batch_size, n_iter=100):
    index = range(length)
    for j in xrange(n_iter):
        np.random.shuffle(index)
        for i in xrange(int(length / batch_size)):
            yield index[i * batch_size:(i + 1) * batch_size]


def load_w2v(w2v_file, embedding_dim, debug=False):
    fp = open(w2v_file)
    fp.readline()

    w2v = []
    word_dict = dict()
    vocabulary = []
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    print 'loading word_embedding {}...'.format(w2v_file)
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:
            print 'a bad word embedding: {}'.format(line[0])
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
        vocabulary.append(line[0])
    print 'done!'
    w2v = np.asarray(w2v, dtype=np.float32)
    # w2v -= np.mean(w2v, axis = 0) # zero-center
    #w2v /= np.std(w2v, axis = 0)
    if debug:
        print 'shape of w2v:', np.shape(w2v)
        print 'id of \'the\':', word_dict['的']
        print 'vector of \'the\':', w2v[word_dict['的']]
    # print vocabulary[0]
    # print len(vocabulary)
    return word_dict, w2v, vocabulary


# def change_y_to_onehot(y, n_class=5):
#     onehot = []
#     for label in y:
#         tmp = [0] * n_class
#         tmp[label] = 1
#         onehot.append(tmp)
#     return np.asarray(onehot, dtype=np.int32)


# def load_data_for_Emotion_CNN(input_file, word_to_id, max_doc_len=10, index=11, n_class=8):
#     x1, y1, doc_len1 = [], [], []
#     x2, y2, doc_len2 = [], [], []
#     x3, y3, doc_len3 = [], [], []

#     label_to_int = dict()
#     labels = ['Joy', 'Hate', 'Love', 'Sorrow',
#               'Anxiety', 'Surprise', 'Anger', 'Expect']
#     for i in range(len(labels)):
#         label_to_int[labels[i]] = i

#     print 'loading input {}...'.format(input_file)
#     if index > 10:
#         index = random.randint(1, 10)

#     for line in open(input_file):
#         line = line.split('\t\t')
#         wordlist = line[-1].split()

#         tmp_x = np.zeros((max_doc_len), dtype=np.int)
#         i = 0
#         for word in wordlist:
#             if i >= max_doc_len:
#                 break
#             if word in word_to_id:
#                 tmp_x[i] = word_to_id[word]
#                 i += 1

#         for j in range(len(line) - 1):
#             if line[j] not in label_to_int:
#                 print 'error:', line[j]
#             else:
#                 line[j] = label_to_int[line[j]]
#                 if index % 5 == 0:
#                     x2.append(tmp_x)
#                     y2.append(line[j])
#                     doc_len2.append(i)
#                     if j == 0:
#                         tmp = [0] * n_class
#                         x3.append(tmp_x)
#                         y3.append(tmp)
#                         doc_len3.append(i)
#                     y3[-1][line[j]] = 1
#                 else:
#                     x1.append(tmp_x)
#                     y1.append(line[j])
#                     doc_len1.append(i)

#         index += 1

#     y1 = change_y_to_onehot(y1, n_class)
#     y2 = change_y_to_onehot(y2, n_class)
#     print 'done!'

#     x1, y1, doc_len1 = np.asarray(x1), np.asarray(y1), np.asarray(doc_len1)
#     x2, y2, doc_len2 = np.asarray(x2), np.asarray(y2), np.asarray(doc_len2)
#     x3, y3, doc_len3 = np.asarray(x3), np.asarray(y3), np.asarray(doc_len3)
#     return x1, y1, doc_len1, x2, y2, doc_len2, x3, y3, doc_len3


def load_data_for_Emotion_CNN_prob(input_file, word_to_id, max_doc_len=10, index=11):
    x1, y_sentiment, y_emotion, doc_len1 = [], [], [], []
    # x2, y2, doc_len2 = [], [], []

    print 'loading input {}...'.format(input_file)
    for line in open(input_file):
        line = line.split('\t\t')
        wordlist = line[-1].split()

        tmp_x = np.zeros((max_doc_len), dtype=np.int)
        i = 0
        for word in wordlist:
            if i >= max_doc_len:
                break
            if word in word_to_id:
                tmp_x[i] = word_to_id[word]
                i += 1
                
        #将标签分数归一化
        #sentiment
        tmp_ys = map(float, line[:2])
        sumys = sum(tmp_ys)
        if sumys != 0:
            tmp_ys = [it / sumys for it in tmp_ys]
        #emotion
        tmp_ye = map(float, line[2:10])
        sumye = sum(tmp_ye)
        if sumye != 0:
            tmp_ye = [it1 / sumye for it1 in tmp_ye]

        #句子长度为i,index 为 test_index

        if index % 10 != 0:
            # x2.append(tmp_x)
            # y2.append(tmp_y)
            # doc_len2.append(i)
            # x1.append(tmp_x)
            #5-grams
            for j in range(int(np.ceil(i/5))):
                start = (j*5)%i
                end = min(start + 5, i)
                x1.append(tmp_x[start:end])
                y_sentiment.append(tmp_ys)
                y_emotion.append(tmp_ye)
                doc_len1.append(end - start)

        index += 1

    print 'done!'

    x1, y_sentiment, y_emotion, doc_len1 = np.asarray(x1), \
    np.asarray(y_sentiment),  np.asarray(y_emotion), np.asarray(doc_len1)
    # x2, y2, doc_len2 = np.asarray(x2), np.asarray(y2), np.asarray(doc_len2)
    return x1, y_sentiment, y_emotion, doc_len1


if __name__ == '__main__':
    print np.ceil(4.1)