# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold,StratifiedKFold

from keras import metrics

from sklearn.svm import SVC

#兼容在没有显示器的GPU服务器上运行该代码
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from keras.utils import plot_model



#yelp评论文件路径 已经使用https://github.com/Yelp/dataset-examples处理成CSV格式
#yelp_file="/Volumes/maidou/dataset/yelp/dataset/review.csv"

yelp_file="/mnt/nlp/dataset/review.csv"


#词袋模型的最大特征束
max_features=5000


def load_reviews(filename):
    #CSV格式表头内容：
    # funny,user_id,review_id,text,business_id,stars,date,useful,cool
    text=[]
    stars=[]

    #https://www.cnblogs.com/datablog/p/6127000.html
    #sep : str, default ‘,’指定分隔符。如果不指定参数，则会尝试使用逗号分隔。分隔符长于一个字符并且不是‘\s+’,将使用python的语法分析器。
    # 并且忽略数据中的逗号。正则表达式例子：'\r\t'
    #header: int or list of ints, default ‘infer’
    # 指定行数用来作为列名，数据开始行数。如果文件中没有列名，设置为None。设置为0则认为第0行是列名
    #nrows : int, default None 需要读取的行数（从文件头开始算起）。
    #skiprows : list-like or integer, default None 需要忽略的行数（从文件开始处算起），或需要跳过的行号列表（从0开始）。
    #skip_blank_lines : boolean, default True如果为True，则跳过空行；否则记为NaN。

    ###
    #开发阶段读取前10000行
    df = pd.read_csv(filename,sep=',',header=0,nrows=10000)
    print df.head()

    #按照列名直接获取数据 把 list转换成list对象
    text=list(df['text'])
    stars=list(df['stars'])

    #显示各个评分的个数
    print df.describe()

    #绘图
    plt.figure()
    count_classes=pd.value_counts(df['stars'],sort=True)

    print "各个star的总数:"
    print count_classes
    count_classes.plot(kind='bar',rot=0)
    plt.xlabel('stars')
    plt.ylabel('stars counts')
    #plt.show()
    plt.savefig("yelp_stars.png")


    return text,stars

#实用SVM进行文档分类
def do_svm(text,stars):
    # 切割词袋 删除英文停用词
    #vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=max_features,stop_words='english',lowercase=True)
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=max_features, stop_words='english', lowercase=True)
    #vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=5000, stop_words=None, lowercase=True)

    print "vectorizer 参数:"
    print vectorizer
    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 使用2-gram和TFIDF处理
    x = transformer.fit_transform(vectorizer.fit_transform(text))
    #x = vectorizer.fit_transform(text)

    #二分类 标签直接实用stars
    y=stars

    clf = SVC()

    # 使用5折交叉验证
    scores = cross_val_score(clf, x, y, cv=5, scoring='f1_micro')
    # print scores
    print("f1_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#使用keras的MLP
def do_keras_mlp(text,stars):
    # 切割词袋 删除英文停用词
    #vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=max_features,stop_words='english',lowercase=True)
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=max_features, stop_words='english', lowercase=True)
    #vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=5000, stop_words=None, lowercase=True)

    print "vectorizer 参数:"
    print vectorizer
    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 使用2-gram和TFIDF处理
    x = transformer.fit_transform(vectorizer.fit_transform(text))
    #x = vectorizer.fit_transform(text)

    #我们可以使用从scikit-learn LabelEncoder类。
    # 这个类通过 fit() 函数获取整个数据集模型所需的编码,然后使用transform()函数应用编码来创建一个新的输出变量。
    encoder=LabelEncoder()
    encoder.fit(stars)
    encoded_y = encoder.transform(stars)

    #构造神经网络
    def baseline_model():
        model = Sequential()
        model.add(Dense(5, input_dim=max_features, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #可视化
        #plot_model(model, to_file='yelp-mlp-model.png',show_shapes=True)

        #model.summary()

        return model
    #在 scikit-learn 中使用 Keras 的模型,我们必须使用 KerasClassifier 进行包装。这个类起到创建并返回我们的神经网络模型的作用。
    # 它需要传入调用 fit()所需要的参数,比如迭代次数和批处理大小。
    # 最新接口指定训练的次数为epochs
    clf = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=128, verbose=0)

    #使用5折交叉验证
    scores = cross_val_score(clf, x, encoded_y, cv=5, scoring='f1_micro')
    # print scores
    print("f1_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #scores = cross_val_score(clf, x, encoded_y, cv=5, scoring='accuracy')
    # print scores
    #print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':

    text,stars=load_reviews(yelp_file)

    stars=[ 0 if star < 3 else 1 for star in stars ]

    print "情感分类的总数:"
    count_classes = pd.value_counts(stars, sort=True)
    print count_classes
    count_classes.plot(kind='bar',rot=0)
    plt.xlabel('sentiment ')
    plt.ylabel('sentiment  counts')
    #plt.show()
    plt.savefig("yelp_sentiment_stars.png")


    #使用MLP文档分类
    #do_keras_mlp(text,stars)
    #使用SVM文档分类
    do_svm(text,stars)