# -*- coding: utf-8 -*-


import os
import jieba
import numpy as np
import math


DATA_DIR = '/Users/wangjipeng/Tensorflow/I\'ll make you regret it/CNN_Text/data/sohu_news_data'
DATA_DIR_WRITE = '/Users/wangjipeng/Tensorflow/I\'ll make you regret it/CNN_Text/data/jieba_sohu_news_data'
STOPWORDS_DIR = '/Users/wangjipeng/Tensorflow/I\'ll make you regret it/CNN_Text/data/sohu_news_stopwords.txt'

def init_sohu_data():
    lis_dir = os.listdir(DATA_DIR)
    stopwords_set = init_stopwords()
    for folder_name in lis_dir:
        folder_lis = os.path.join(DATA_DIR, folder_name)
        all_file_lis = os.listdir(folder_lis)
        for file_name in all_file_lis:
            file_path = os.path.join(folder_lis, file_name)
            file_path_write_name = os.path.join(DATA_DIR_WRITE, folder_name, file_name)
            file_path_write = os.path.join(DATA_DIR_WRITE, folder_name)
            if not os.path.exists(file_path_write):
                os.makedirs(file_path_write)
            write_file = open(file_path_write_name, 'w')
            with open(file_path) as reader:
                all_str = reader.read().replace('\n', '').replace('\r', '').strip()
                jieba_str = jieba.cut(all_str)
                word_list = []
                for word in jieba_str:
                    word = word.strip()
                    if '' != word and word not in stopwords_set:
                        word_list.append(word)
                word_str = ' '.join(word_list)
                # print (word_str)
                write_file.write(word_str)



def init_stopwords():
    global sohu_news_stopwords_set

    sohu_news_stopwords_set = set()
    with open(STOPWORDS_DIR, 'r') as reader:
        for each_line in reader.readlines():
                # word = ensure_unicode(each_line.replace('\n', '').strip().lower())
            word = each_line.replace('\n', '').strip().lower()
            # print (word)
            sohu_news_stopwords_set.add(word)
    return sohu_news_stopwords_set



def batch_iter(data, batch_size, shuff=True):

    data = np.array(data)
    len_data = len(data)
    num_epoch = int(math.ceil(float(len(data))/ batch_size))
    if shuff:
        shuff_flag = np.random.permutation(np.arange(len_data))
        shuffled_data = data[shuff_flag]
    else:
        shuffled_data = data
    for batch_num in range(num_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data)
        yield shuffled_data[start_index: end_index]





if __name__ == '__main__':
    init_sohu_data()
    # init_stopwords()