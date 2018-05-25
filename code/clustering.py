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

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import time
from jieba import analyse

from sklearn.metrics import silhouette_score

#测试环境
DEV_FILE="../data/news_sohusite_content_1000.txt"
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
"""
[(0, 0.019423252), (1, 0.019521076), (2, 0.92217809), (3, 0.01954053), (4, 0.019337002)]
这里需要解释的是，无论是词袋模型还是LDA生成的结果，都可能存在大量的0，这会占用大量的内存空间。因此默认情况下，
词袋以及LDA计算的结果都以稀疏矩阵的形式保存。稀疏矩阵的最小单元定义为：

（元素所在的位置，元素的值）
比如一个稀疏矩阵只有0号和2号元素不为0，分别为1和5，那么它的表示方法如下：

[(0,1),(2,5)]
"""
def transformedCorpus2Vec(turples,num_topics):
    ret = [0] * num_topics
    for tuple in turples:
        index = tuple[0]
        weight = tuple[1]
        ret[index] = weight

    return ret


def get_keywords(articles):


    text=""

    text=" ".join(articles)


    keywords_tfidf = analyse.extract_tags(text, topK=10,
                                          withWeight=False, allowPOS=('n', 'ns', 'vn', 'v', 'nz'))

    return keywords_tfidf


def do_cluster(x,articles_content):
    start = time.clock()

    #使用DBSCAN进行聚类分类
    db = DBSCAN(eps=5.8, min_samples=2).fit(x)

    #获取每条记录对应的聚类标签 其中-1表示识别为噪音点
    labels = db.labels_

    #统计聚类个数
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print "n_clusters_:%d" % (n_clusters_)

    #计算耗时
    end = time.clock()
    print('[cluster]Running time: %s Seconds' % (end - start))

    #把列表转成数组np结构
    articles_content=np.array(articles_content)

    for clusters_id in range(1,n_clusters_+1):
        print "/********************************************************************/"
        print "clusters_id %d" % (clusters_id)
        articles=articles_content[labels==clusters_id]

        #keywords=get_keywords(articles)

        #for keyword in keywords:
        #    print keyword + "/"

        for article in articles:
            print article[:200]


def do_cluster_dbscan(x,articles_content):


    for eps in np.arange(0.1,10.0,0.1):

        #使用DBSCAN进行聚类分类
        db = DBSCAN(eps=eps, min_samples=2).fit(x)
        silhouette=silhouette_score(x, db.labels_)

        print "eps=%.2f silhouette=%.2f" % (eps,silhouette)










def do_lda_usemulticore():

    #获取当前时间
    start = time.clock()

    #加载搜狗新闻数据
    content=load_sougou_content()

    #备份原始数据
    articles_content=content

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


    #计算耗时
    end = time.clock()
    print('[lda]Running time: %s Seconds' % (end - start))

    x=lda[texts_tf_idf]

    #格式转化 x是稀疏矩阵 需要转换成正常的格式
    x=[ transformedCorpus2Vec(x,num_topics) for x in x  ]



    #标准化
    x = StandardScaler().fit_transform(x)

    #聚类分析
    #do_cluster_dbscan(x, articles_content)
    do_cluster(x, articles_content)







if __name__ == '__main__':

    #设置NTLK全局停用词
    analyse.set_stop_words("stopwords.txt")

    do_lda_usemulticore()







