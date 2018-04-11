# -*- coding: UTF-8 -*-
import re
from fastText import train_supervised

import numpy as np
import codecs

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

from gensim import corpora, models


def load_stopwords():
    with open("stopwords.txt") as F:
        stopwords=F.readlines()
        F.close()
    return [word.strip() for word in stopwords]


def load_sougou_content():
    #with open("../data/news_sohusite_content.txt") as F:
    with open("../data/news_sohusite_content_10000.txt") as F:
        content=F.readlines()
        F.close()
    return content


if __name__ == '__main__':

    #加载搜狗新闻数据
    content=load_sougou_content()

    #加载停用词
    stopwords=load_stopwords()

    #切割token
    content=[  [word for word in line.split() if word not in stopwords]   for line in content]


    # 得到文档-单词矩阵 （直接利用统计词频得到特征）
    dictionary = corpora.Dictionary(content)

    # 将dictionary转化为一个词袋，得到文档-单词矩阵
    texts = [dictionary.doc2bow(text) for text in content]

    # 利用tf-idf来做为特征进行处理
    texts_tf_idf = models.TfidfModel(texts)[texts]


    # 利用LDA做主题分类的情况
    print "LDA"

    num_topics=6

    lda = models.ldamodel.LdaModel(corpus=texts, id2word=dictionary, num_topics=num_topics)

    #print lda.print_topics(num_topics=num_topics, num_words=4)

    for topic in lda.print_topics(num_topics=num_topics, num_words=10):
        print topic[1]






