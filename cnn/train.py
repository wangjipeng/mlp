# -*- coding: utf-8 -*-
'''
2018，06.04

'''


import argparse
import os
import time
import numpy as np

from  tensorflow.contrib import learn
import tensorflow as tf

from preData import init_sohu_data, DATA_DIR_WRITE
import TextCnn

def initParameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./sohu/', help='the date dir')
    parser.add_argument('--model-dir', type=str, default='./model', help='The path of saved model')
    parser.add_argument('--model-name', type=str, default='text_cnn', help='Model name')

    parser.add_argument('--max-vocabulary', type=int, default=1000, help='the max vocabulary')
    parser.add_argument('--valid-num', type=int, default=1000, help='Number of validation')
    parser.add_argument('--max-textlength', type=int, default=1000, help='the max textlength')
    parser.add_argument('--classnum', type=int, default=10, help='the class num')
    parser.add_argument('--nnkind', type=str, default='cnn', help='the nn kinds')

    parser.add_argument('--filterSizes', type=str, default='3,4,5', help='Comma-separated filter sizes')
    parser.add_argument('--filterNum', type=int, default=100, help='the file nums')
    parser.add_argument('--embedingNum', type=int, default=60, help= 'the vocabulary size')
    parser.add_argument('--acctive', type=str, default='relu', help = 'the acctive ')
    parser.add_argument('--optimizer', type=str, default='relu', help='Activation function')
    parser.add_argument('--l2-reg-lambda', type=float, default=0.0, help='L2 regularization lambda')
    parser.add_argument('--init-scale', type=float, default=0.1, help='Init scale')



    config = parser.parse_args()

    print (config.max_vocabulary)
    print (config.max_textlength)
    print (config.classnum)
    print (config.nnkind)
    print (config.filterSizes)
    print (config.filterNum)
    print (config.embedingNum)
    print (config.acctive)
    print (config.optimizer)



    return config


class Train(object):
    def __init__(self, config):
        self.config = config
        with tf.name_scope('valid') as scope:
            with tf.variable_scope('model', reuse=True):
                self.input_x = tf.placeholder(name='input_x', dtype=tf.float32, shape=[None, config.max_textlength])
                self.output_y = tf.placeholder(name='output_y', dtype=tf.float32, shape=[None, config.classNum])
                self.dropout = tf.placeholder(name='dropout', dtype=tf.float32)

                if 'text_cnn' == config.kind:
                    self.model = TextCnn(self.input_x, self.output_y, self.dropout, self.config, scope)
                elif 'text_rnn' == config.kind:
                    print 'text_rnn'
                else:
                    print ('not TextCnn or TextRnn ...')
                    exit(0)





def main():
    # init parameter
    print('Loading config...')
    config = initParameter()
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(config.model_dir, config.model_name, timestamp))
    if os.path.exists(out_dir):
        os.mkdirs(out_dir)
    print('Writing to {}\n'.format(out_dir))
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    # init data
    print ('Initing data...')
    if not os.path.exists(DATA_DIR_WRITE):
        init_sohu_data()
    label_value = os.listdir(DATA_DIR_WRITE)
    X = []
    y = []
    label_index = { v: i for i, v in enumerate(label_value)}
    for label in label_value:
        label_path = os.path.join(DATA_DIR_WRITE, label)
        label_item = np.zeros(len(label_value), np.float32)
        label_item[label_index[label]] = 1
        for file_name in os.listdir(label_path):
            file_path = os.path.join(DATA_DIR_WRITE, label, file_name)
            with open(file_path, 'r') as reader:
                context = reader.read().replace('\n', '').replace('\r', '').strip()
                X.append(context)
                y.append(label_item)
    Y = np.array(y)

    # init Vocabulary
    print ('Initing Vocabulary...')
    Vocabulary = learn.preprocessing.VocabularyProcessor(config.max_vocabulary)
    X = np.array(list(Vocabulary.fit_transform(X)))
    print (len(X))

    # init shuffle data
    print ('Initing shuffle data')
    np.random.seed(0)
    shuffle_index = np.arange(len(y))
    x_shuffled = X[shuffle_index]
    y_shuffled = Y[shuffle_index]
    valid_sample_index = -config.valid_num
    x_train, x_valid = x_shuffled[:valid_sample_index], x_shuffled[valid_sample_index:]
    y_train, y_valid = y_shuffled[:valid_sample_index], y_shuffled[valid_sample_index:]
    print("Vocabulary Size: {:d}".format(len(Vocabulary.vocabulary_)))
    print("Train/Valid split: {:d}/{:d}".format(len(y_train), len(y_valid)))
    config.vocabulary_size = len(Vocabulary.vocabulary_)

    # start the model
    print ('traing the model...')
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    #         train = Train(config)













if __name__ == '__main__':
    main()