# -*- coding: UTF-8 -*-
import re
import os
#from fastText import train_supervised

import numpy as np
import codecs

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#from keras.utils import to_categorical
#from sklearn.preprocessing import OneHotEncoder

from gensim import corpora, models

import time

#测试环境
DEV_FILE="../data/news_sohusite_content_10000.txt"
#生产环境
PRO_FILE="/mnt/nlp/dataset/news_sohusite_content-50000.txt"


def load_stopwords():
    with open("stopwords.txt") as F:
        stopwords=F.readlines()
        F.close()
    return [word.strip() for word in stopwords]


def load_sougou_content():
    filename=PRO_FILE

    if os.path.exists(DEV_FILE):
        filename=DEV_FILE

    print "Open data file %s" % (filename)

    with open(filename) as F:
        content=F.readlines()
        F.close()


    return content

def test_bow():
    content=[
        ["你","爱","我"],["我","爱","她"]
    ]

    test=["你","爱","她"]

    # 得到文档-单词矩阵 （直接利用统计词频得到特征）
    dictionary = corpora.Dictionary(content)

    # 将dictionary转化为一个词袋，得到文档-单词矩阵
    texts = [dictionary.doc2bow(text) for text in content]

    texts=np.array(texts)

    print texts

    test=dictionary.doc2bow(text)

    print test


def do_lda():
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
    #print "BOW&LDA"

    #num_topics=5

    #ldamulticore
    #lda = models.ldamodel.LdaModel(corpus=texts, id2word=dictionary, num_topics=num_topics)
    #lda = models.ldamodel.ldamulticore(corpus=texts, id2word=dictionary, num_topics=num_topics)

    #print lda.print_topics(num_topics=num_topics, num_words=4)

    #打印前5个主题
    #for index,topic in lda.print_topics(5):
    #    print topic

    # 利用TFIDF&LDA做主题分类的情况
    print "TFIDF&LDA"

    num_topics=5

    lda = models.ldamodel.LdaModel(corpus=texts_tf_idf, id2word=dictionary, num_topics=num_topics)

    #print lda.print_topics(num_topics=num_topics, num_words=4)

    #打印前10个主题
    for index,topic in lda.print_topics(5):
        print topic

    #获取预料对应的LDA特征
    corpus_lda = lda[texts_tf_idf]

    #for doc_tfidf in corpus_lda:
    #    print(doc_tfidf)
    print corpus_lda[0]


def do_lda_usemulticore():

    #获取当前时间
    start = time.clock()

    #加载搜狗新闻数据
    content=load_sougou_content()

    #加载停用词
    stopwords=load_stopwords()

    #切割token
    content=[  [word for word in line.split() if word not in stopwords]   for line in content]

    #计算耗时
    end = time.clock()
    print('[data clean]Running time: %s Seconds' % (end - start))


    #获取当前时间
    start = time.clock()


    # 得到文档-单词矩阵 （直接利用统计词频得到特征）
    dictionary = corpora.Dictionary(content)

    # 将dictionary转化为一个词袋，得到文档-单词矩阵
    texts = [dictionary.doc2bow(text) for text in content]

    # 利用tf-idf来做为特征进行处理
    texts_tf_idf = models.TfidfModel(texts)[texts]

    #计算耗时
    end = time.clock()
    print('[get word bag]Running time: %s Seconds' % (end - start))


    # 利用TFIDF&LDA做主题分类的情况
    print "TFIDF&LDA"

    num_topics=200

    start = time.clock()


    #workers指定使用的CPU个数 默认使用cpu_count()-1 即使用几乎全部CPU 仅保留一个CPU不参与LDA计算
    #https://radimrehurek.com/gensim/models/ldamulticore.html
    #Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.
    lda = models.ldamulticore.LdaMulticore(corpus=texts_tf_idf, id2word=dictionary, num_topics=num_topics)
    #lda = models.ldamulticore.LdaMulticore(corpus=texts_tf_idf, id2word=dictionary,
    #                                       num_topics=num_topics,workers=12)

    # 打印前10个主题
    for index, topic in lda.print_topics(num_topics=10, num_words=10):
        print topic


    #计算耗时
    end = time.clock()
    print('[lda]Running time: %s Seconds' % (end - start))

if __name__ == '__main__':

    #test_bow()
    #do_lda()
    do_lda_usemulticore()







