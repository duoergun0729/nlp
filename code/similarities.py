# -*- coding: UTF-8 -*-
import re
from fastText import train_supervised

import numpy as np
import codecs
import sys

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


from gensim import corpora, models,similarities

from simhash import Simhash
import math


def load_stopwords():
    with open("stopwords.txt") as F:
        stopwords=F.readlines()
        F.close()
    return [word.strip() for word in stopwords]


def load_sougou_content():
    #with open("../data/news_sohusite_content.txt") as F:
    # 测试阶段仅加载前1w条记录
    with open("../data/news_sohusite_content_10000.txt") as F:
        content=F.readlines()
        F.close()
    return content


def gensim_sim(content,test_news):
    # 加载积累的stopwords
    stopwords = load_stopwords()

    # 切割token并清除stopwords
    x = [[word for word in line.split() if word not in stopwords] for line in content]

    # 获取词袋
    dictionary = corpora.Dictionary(x)

    # 制作语料
    corpus = [dictionary.doc2bow(doc) for doc in x]

    # 进行TFIDF处理
    tfidf = models.TfidfModel(corpus)

    # 把测试文章转换成tfidf
    test_news_vec = [word for word in test_news.split() if word not in stopwords]

    test_news_vec = tfidf[dictionary.doc2bow(test_news_vec)]

    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
    sim = index[tfidf[test_news_vec]]

    # for index, similarities in sorted(enumerate(sim), key=lambda item: -item[1])[:4]:
    # print   "index:%d similarities:%d content:%s" % ( index, similarities,content[index] )
    #    print   "index:%d similarities:%d" % (index, similarities)
    for index, score in sorted(enumerate(sim), key=lambda item: -item[1])[:6]:
        #print   "index:%d similarities:%f" % (index, score)
        print   "index:%d similarities:%f content:%s" % (index, score, content[index])


def gensim_simhash(content,test_news):

    # 加载积累的stopwords
    stopwords = load_stopwords()

    # 切割token并清除stopwords
    x = [[word for word in line.split() if word not in stopwords] for line in content]

    # 切割token并清除stopwords
    test_news = [word for word in test_news.split() if word not in stopwords]

    # 计算simhash
    test_news_hash = Simhash(test_news)


    sim=[]
    # 遍历语料计算simhash值
    for news in x:
        hash = Simhash(news)
        score=test_news_hash.distance(hash)
        sim.append( score)
        #print "add %d %f" %(index,score)

    for index, score in sorted(enumerate(sim), key=lambda item: item[1])[:6]:
        # print   "index:%d similarities:%f" % (index, score)
        print   "index:%d similarities:%f content:%s" % (index, score, content[index])


def gensim_cos(content,test_news):
    # 加载积累的stopwords
    stopwords = load_stopwords()

    # 切割token并清除stopwords
    x = [[word for word in line.split() if word not in stopwords] for line in content]

    # 获取词袋
    dictionary = corpora.Dictionary(x)

    # 制作语料
    corpus = [dictionary.doc2bow(doc) for doc in x]

    # 把测试文章转换成tfidf
    test_news_vec = [word for word in test_news.split() if word not in stopwords]

    test_news_vec = dictionary.doc2bow(test_news_vec)


    #稀疏矩阵转正常矩阵
    #corpus=corpus.toarray()
    #test_news_vec=test_news_vec.toarray()

    cos=[]
    # 遍历语料计算余弦值值
    for news in corpus:
        a=np.array(news)
        b=np.array(test_news_vec)
        c=np.dot(a,b)
        score=c/(math.sqrt(a)*math.sqrt(b))
        cos.append( score)

    for index, score in sorted(enumerate(cos), key=lambda item: item[1])[:6]:
        # print   "index:%d similarities:%f" % (index, score)
        print   "index:%d similarities:%f content:%s" % (index, score, content[index])

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')


    #加载搜狐新闻语料
    content=load_sougou_content()


    #设置测试文章
    print "select test data:"
    test_news=content[88]
    print test_news

    #gensim_sim(content, test_news)

    #print "simhash"
    #gensim_simhash(content, test_news)

    print "cos"
    gensim_cos(content, test_news)

