# -*- coding: utf-8 -*-
'''
2018, 06, 07
'''


import tensorflow as tf


def print_variable_info(var):
    """
    print variable info
    """
    print (var.op.name, ' ', var.get_shape().as_list())


class TextCnn(object):

    def __init__(self, input_x, output_y, dropout, config, scope):
        with tf.variable_scope('input_layer'):
            self.input_x = input_x
            self.output_y = output_y
            self.classNum = config.classnum
            self.maxLength = config.max_textlength
            self.filterSizes = [ int(x) for x in config.filterSizes.split(',')]
            self.filterNum = config.filterNum
            self.embedingNum = config.embedingNum
            self.vocabulary_size = config.vocabulary_size
            if 'relu' == config.acctive:
                self.acctive = tf.nn.relu
            else:
                self.activation = tf.nn.tanh
            if 'adam' == config.optimizer:
                self.optimizer = tf.train.AdamOptimizer
            else:
                self.optimizer = tf.train.GradientDescentOptimizer
            self.l2_reg_lambda = config.l2_reg_lambda
            # L2正规化损失记录（可选）
            l2_loss = tf.constant(0.0)
            self.init_scale = config.init_scale
        with tf.variable_scope('embeding_layer'):
            self.embedding_weight = tf.get_variable(name='weight', type=tf.float32, shape=[self.vocabulary_size, self.embedingNum],
                                          initializer=tf.random_uniform_initializer(-1 * self.init_scale,
                                                                                    1 * self.init_scale))
            print_variable_info(self.embedding_weight)
            self.look_table = tf.nn.embedding_lookup(self.embedding_weight, self.input_x)
            print_variable_info(self.look_table)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        maxpool_lis = []
        for i, filter_size in enumerate(self.filterSizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                con_weight = tf.get_variable(name='weight', dtype=tf.float32, shape=[filter_size, self.embedingNum, 1, self.filterNum], initializer=tf.truncated_normal_initializer(stddev=0.1))
                con_baise = tf.get_variable(name='baise', dtype=tf.float32, shape=[self.filterNum], initializer=tf.constant_initializer(0.1))
                self.con2d = tf.nn.conv2d(self.embedded_chars_expanded, con_weight, strides=[1,1,1,1], name='conv2d', padding="VALID")
                h = self.acctive(tf.nn.bias_add(self.con2d, con_baise), name='activation')
                print_variable_info(h)
                pooled = tf.nn.pool(h,
                        ksize=[1, self.sentence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool",
                )
                maxpool_lis.append(pooled)




