#!/usr/bin/env python
# encoding: utf-8


import sys
import time
import os
import numpy as np
import tensorflow as tf
from PrepareData import batch_index, load_w2v, load_data_for_Emotion_CNN_prob
# from Evalue import Emotion_eval

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 200, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.0001, 'l2 regularization')

tf.app.flags.DEFINE_integer('display_step', 1, 'number of test display step')
tf.app.flags.DEFINE_integer('training_iter', 10, 'number of train iter')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('n_class_sentiment', 2, 'number of distinct sentiment class')
tf.app.flags.DEFINE_integer('n_class_emotion', 8, 'number of distinct emotion class')
tf.app.flags.DEFINE_integer('max_doc_len', 5, 'max number of tokens per sentence')

tf.app.flags.DEFINE_string('train_file_path', 'renCECps_sentiment_emotion_score.txt', 'training file')
tf.app.flags.DEFINE_string('embedding_file_path', 'vectors_iter15.txt', 'embedding file')
tf.app.flags.DEFINE_string('test_index', 1, 'test_index')
tf.app.flags.DEFINE_string('embedding_type', 0, 'embedding_type')

class Emotion_CNN(object):

    def __init__(self,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 keep_prob=FLAGS.keep_prob,
                 l2_reg=FLAGS.l2_reg,
                 display_step=FLAGS.display_step,
                 training_iter=FLAGS.training_iter,
                 embedding_dim=FLAGS.embedding_dim,
                #  n_class=FLAGS.n_class,
                 n_class_sentiment = FLAGS.n_class_sentiment,
                 n_class_emotion = FLAGS.n_class_emotion,
                 max_doc_len=FLAGS.max_doc_len,
                 train_file_path=FLAGS.train_file_path,
                 w2v_file=FLAGS.embedding_file_path,
                 test_index=FLAGS.test_index,
                 embedding_type=FLAGS.embedding_type,
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.Keep_Prob = keep_prob
        self.l2_reg = l2_reg

        self.display_step = display_step
        self.training_iter = training_iter
        self.embedding_dim = embedding_dim
        # self.n_class = n_class
        self.n_class_sentiment = n_class_sentiment
        self.n_class_emotion = n_class_emotion

        self.max_doc_len = max_doc_len

        self.train_file_path = train_file_path
        self.w2v_file = w2v_file
        self.test_index = test_index
        self.embedding_type = embedding_type

        self.word_id_mapping, self.w2v, self.vocabulary = load_w2v(
            self.w2v_file, self.embedding_dim)

        if embedding_type == 0:  # Pretrained and Untrainable
            self.word_embedding = tf.constant(
                self.w2v, dtype=tf.float32, name='word_embedding')
        elif embedding_type == 1:  # Pretrained and Trainable
            self.word_embedding = tf.Variable(
                self.w2v, dtype=tf.float32, name='word_embedding')
        elif embedding_type == 2:  # Random and Trainable
            self.word_embedding = tf.Variable(tf.random_uniform(
                [len(self.word_id_mapping) + 1, self.embedding_dim], -0.1, 0.1), name='word_embedding')

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_doc_len])
            # self.y = tf.placeholder(tf.float32, [None, self.n_class])
            self.y_sentiment = tf.placeholder(tf.float32, [None, self.n_class_sentiment])
            self.y_emotion = tf.placeholder(tf.float32, [None, self.n_class_emotion])
            self.doc_len = tf.placeholder(tf.int32, None)
            self.keep_prob = tf.placeholder(tf.float32)

        def init_variable(shape):
            initial = tf.random_uniform(shape, -0.01, 0.01)
            return tf.Variable(initial)

        with tf.name_scope('weights'):
            self.weights = {
                'conv1': init_variable([5, self.embedding_dim, 1, 300]),
                'softmax1': init_variable([300, self.n_class_sentiment]),
                'softmax2': init_variable([300, self.n_class_emotion]),                

            }

        with tf.name_scope('biases'):
            self.biases = {
                'conv1': init_variable([300]),
                # 'conv2': init_variable([100]),
                # 'conv3': init_variable([100]),
                'softmax1': init_variable([self.n_class_sentiment]),
                'softmax2': init_variable([self.n_class_emotion]),                

            }

    def model(self, inputs):

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

        # def max_over_time_pooling(x, step):
        #     return tf.nn.max_pool(x, ksize=[1, step, 1, 1], strides=[1, 3, 1, 1], padding='VALID')

        def AcFun(x):
            return tf.nn.relu(x)

        def conv_pooling(inputs, step, w, b):
            conv = conv2d(inputs, w) + b
            h_conv = AcFun(conv)
            output = tf.reshape(h_conv, [-1,300])

            # h_pool = max_over_time_pooling(h_conv, step)
            # output = tf.reshape(h_pool, [-1,100])
            return output

        inputs = tf.reshape(inputs, [-1, self.max_doc_len, self.embedding_dim, 1])

        with tf.name_scope('sentence_conv_pooling'):
            outputs1 = conv_pooling(inputs, 1, self.weights['conv1'], self.biases['conv1'])
            outputs = outputs1
            # outputs2 = conv_pooling(inputs, self.max_doc_len-2, self.weights['conv2'], self.biases['conv2'])
            # outputs3 = conv_pooling(inputs, self.max_doc_len-3, self.weights['conv3'], self.biases['conv3'])
            # outputs = tf.concat(1, (outputs1, outputs2, outputs3))

        with tf.name_scope('softmax1'):
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
            predict_sentiment = tf.matmul(outputs, self.weights['softmax1']) + self.biases['softmax1']
            predict_sentiment = tf.nn.softmax(predict_sentiment)

        with tf.name_scope('softmax2'):
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
            predict_emotion = tf.matmul(outputs, self.weights['softmax2']) + self.biases['softmax2']
            predict_emotion = tf.nn.softmax(predict_emotion)

        return predict_sentiment, predict_emotion



    def run(self):
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        prob_sentiment, prob_emotion = self.model(inputs)

        with tf.name_scope('loss'):
            # cost = - tf.reduce_mean(self.y * tf.log(prob))
            # 联合损失函数
            lamda_emotion = 0.7
            cost = - (1- lamda_emotion) * tf.reduce_mean(self.y_sentiment * tf.log(prob_sentiment))\
             - lamda_emotion * tf.reduce_mean(self.y_emotion * tf.log(prob_emotion))
            reg, variables = tf.nn.l2_loss(self.word_embedding), ['conv1', 'softmax1', 'softmax2']
            for vari in variables:
                reg += tf.nn.l2_loss(self.weights[vari]) + \
                    tf.nn.l2_loss(self.biases[vari])
            cost += reg * self.l2_reg

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        # with tf.name_scope('predict'):
        #     # correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #     correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        # with tf.name_scope('summary'):
        #     localtime = time.strftime("%X %Y-%m-%d", time.localtime())
        #     Summary_dir = 'Summary/' + localtime

        #     info = 'batch-{}, lr-{}, kb-{}, l2_reg-{}'.format(
        #         self.batch_size,  self.learning_rate, self.Keep_Prob, self.l2_reg)
        #     info = info + '\ntrain_file_path:' + self.train_file_path + '\ntest_index:' + str(self.test_index) + '\nembedding_type:' + str(self.embedding_type) + '\nMethod: Emotion_CNN'
        #     summary_acc = tf.scalar_summary('ACC ' + info, accuracy)
        #     summary_loss = tf.scalar_summary('LOSS ' + info, cost)
        #     summary_op = tf.merge_summary([summary_loss, summary_acc])

        #     test_acc = tf.placeholder(tf.float32)
        #     test_loss = tf.placeholder(tf.float32)
        #     summary_test_acc = tf.scalar_summary('ACC ' + info, test_acc)
        #     summary_test_loss = tf.scalar_summary('LOSS ' + info, test_loss)
        #     summary_test = tf.merge_summary(
        #         [summary_test_loss, summary_test_acc])

        #     train_summary_writer = tf.train.SummaryWriter(
        #         Summary_dir + '/train')
        #     test_summary_writer = tf.train.SummaryWriter(Summary_dir + '/test')

        # with tf.name_scope('saveModel'):
        #     saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        #     save_dir = 'Models/' + localtime + '/'
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)

        with tf.name_scope('readData'):
            print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
            # tr_x, tr_y, tr_doc_len, te_x, te_y, te_doc_len, ev_x, ev_y, ev_doc_len= load_data_for_Emotion_CNN(
            #     self.train_file_path,
            #     self.word_id_mapping,
            #     self.max_doc_len,
            #     self.test_index,
            #     self.n_class
            # )
            tr_x, tr_y_sentiment, tr_y_emotion, tr_doc_len = load_data_for_Emotion_CNN_prob(
                self.train_file_path,
                self.word_id_mapping,
                self.max_doc_len,
                self.test_index,
                # self.n_class
            )
            print 'train docs_sentiment: {} docs_emotion: {}'.format(len(tr_y_sentiment), len(tr_y_emotion))
            print 'training_iter:', self.training_iter
            # print info
            print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            max_acc, bestIter = 0., 0

            # def test():
            #     acc, loss, cnt = 0., 0., 0
            #     for test, num in self.get_batch_data(te_x, te_y, te_doc_len, 20, keep_prob=1.0):
            #         _loss, _acc = sess.run([cost, correct_num], feed_dict=test)
            #         acc += _acc
            #         loss += _loss * num
            #         cnt += num
            #     loss = loss / cnt
            #     acc = acc / cnt
            #     return loss, acc

            # def new_test():
            #     feed_dict = {
            #         self.x: ev_x,
            #         self.doc_len: ev_doc_len,
            #         self.keep_prob: 1.0,
            #     }
            #     y_true = ev_y
            #     y_pred_p = sess.run(prob, feed_dict=feed_dict)
            #     #修改 ： 采用calibrated label ranking 算法
            #     # y_pred = np.ceil(y_pred_p-1.0/8)
            #     y_pred  = calibrated_label_ranking(y_pred_p)
            #     Emotion_eval(y_true, y_pred, y_pred_p)

            # if self.training_iter==0:
            #     saver.restore(sess, 'Models/10:01:44 2017-03-11/-856')
            #     loss, acc=test()
            #     print loss,acc
            #     new_test()

            for i in xrange(self.training_iter):

                for train, _ in self.get_batch_data(tr_x, tr_y_sentiment, tr_y_emotion, tr_doc_len, self.batch_size, self.Keep_Prob):
                    _, step,  loss = sess.run(
                        [optimizer, global_step, cost], feed_dict=train)
                    # train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}'.format(step, loss)

                # if i % self.display_step == 0:
                #     loss, acc=test()

                    # if acc > max_acc:
                    #     max_acc = acc
                    #     bestIter = step
                    #     saver.save(sess, save_dir, global_step=step)
                    #     new_test()

                    # summary = sess.run(summary_test, feed_dict={
                    #                    test_loss: loss, test_acc: acc})
                    # test_summary_writer.add_summary(summary, step)
                    # print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    # print 'Iter {}: test loss={:.6f}, test acc={:.6f}'.format(step, loss, acc)
                    # print 'round {}: max_acc={} BestIter={}\n'.format(i, max_acc, bestIter)

            print 'Optimization Finished!'

            #训练好的embedding

            print sess.run(self.word_embedding)
            print self.word_embedding
            fw_embedding = open('multi-task_embedding_result','w')
            # fw_embedding.write(sess.run(self.word_embedding))
            print>> fw_embedding, "43593 200"

            index = 0
            temp_embedding = (sess.run(self.word_embedding))[1:]
            print len(temp_embedding)
            for item in temp_embedding:
                #w2v[word_dict['的']]
                string = self.vocabulary[index] + " "
                for it in item:
                    string += str(round(it ,6)) + " "
                print>> fw_embedding, string
                index += 1


    def get_batch_data(self, x, y_sentiment, y_emotion, doc_len, batch_size, keep_prob):
        for index in batch_index(len(y_sentiment), batch_size, 1):
            feed_dict = {
                self.x: x[index],
                self.y_sentiment: y_sentiment[index],
                self.y_emotion: y_emotion[index],                
                self.doc_len: doc_len[index],
                self.keep_prob: keep_prob,
            }
            yield feed_dict, len(index)



def main(_):
    for i in [1]: #Pretrained and Trainable
        for j in [0]:
            print('hhhe_ebdt{}_index{}.txt'.format(i, j))
            #sys.stdout = open('ebdt'+str(i)+'_index'+str(j)+'.txt', 'w')
            obj = Emotion_CNN(
               test_index=j,
               embedding_type=i,
            )
            obj.run()

            
    
if __name__ == '__main__':
    tf.app.run()