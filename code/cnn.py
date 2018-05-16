#coding=utf-8
import pandas as pd
import numpy as np
import sys

#处理编码问题
reload(sys)
sys.setdefaultencoding('utf-8')

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

from keras.layers import Embedding, LSTM

import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer
from  keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from fastText import train_supervised
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import wordnet
import enchant

from keras.layers import Conv1D,GlobalMaxPooling1D,Activation,Input,MaxPooling1D,Flatten,concatenate,Embedding

from keras.models import Model

from gensim import corpora, models,similarities
from gensim.models import word2vec,KeyedVectors






#兼容在没有显示器的GPU服务器上运行该代码
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from keras.utils import plot_model



#yelp评论文件路径 已经使用https://github.com/Yelp/dataset-examples处理成CSV格式
#yelp_file="/Volumes/maidou/dataset/yelp/dataset/review.csv"
yelp_file="/mnt/nlp/dataset/review.csv"
#word2vec_file="/Volumes/maidou/dataset/gensim/GoogleNews-vectors-negative300.bin"
word2vec_file="/mnt/nlp/dataset/GoogleNews-vectors-negative300.bin"

#词袋模型的最大特征束
max_features=5000


def load_reviews(filename,nrows):
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
    #开发阶段读取前10000行 使用encoding='utf-8'参数非常重要
    df = pd.read_csv(filename,sep=',',header=0,encoding='utf-8',nrows=nrows)
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

def pad_sentences(data,maxlen=56,values=0.,vec_size = 300):
    """padding to max length
    :param data:要扩展的数据集
    :param maxlen:扩展的h长度
    :param values:默认的值
    """
    length = len(data)
    if length < maxlen:
        for i in range(maxlen - length):
            data.append(np.array([values]*vec_size))
    return data


#使用词向量表征英语句子
def get_vec_by_sentence_list(word_vecs,sentence_list,maxlen=56,vec_size = 300):
    data = []
    values=0.0
    for sentence in sentence_list:
        # get a sentence
        sentence_vec = []
        words = sentence.split()
        for word in words:

            try:
                sentence_vec.append(word_vecs[word].tolist())
            except:
                print word

        # padding sentence vector to maxlen(w * h)
        sentence_vec = pad_sentences(sentence_vec,maxlen,values,vec_size)
        # add a sentence vector
        data.append(np.array(sentence_vec))
    return data



#使用keras的单层cnn
def do_keras_cnn(text,stars):

    #转换成词袋序列
    max_document_length=200



    #设置分词最大个数 即词袋的单词个数
    tokenizer = Tokenizer(num_words=max_features,lower=True)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    x=pad_sequences(sequences, maxlen=max_document_length)


    #print "加载GoogleNews-vectors-negative300.bin..."
    #model = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    #print "加载完毕"

    #print model['boy'].shape

    #词向量的维数 GoogleNews-vectors-negative300.bin维数为300
    #max_features=300

    #x = np.concatenate([buildWordVector(model, z, 50) for z in text])
    #x = get_vec_by_sentence_list(model,text,max_document_length,max_features)



    #我们可以使用从scikit-learn LabelEncoder类。
    # 这个类通过 fit() 函数获取整个数据集模型所需的编码,然后使用transform()函数应用编码来创建一个新的输出变量。
    encoder=LabelEncoder()
    encoder.fit(stars)
    encoded_y = encoder.transform(stars)



    #构造神经网络
    def baseline_model():

        #CNN参数
        embedding_dims = 50
        filters = 250
        kernel_size = 3
        hidden_dims = 250

        model = Sequential()
        model.add(Embedding(max_features, embedding_dims))

        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        #池化
        model.add(GlobalMaxPooling1D())

        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        #可视化
        plot_model(model, to_file='yelp-cnn-model.png',show_shapes=True)

        model.summary()

        return model
    #在 scikit-learn 中使用 Keras 的模型,我们必须使用 KerasClassifier 进行包装。这个类起到创建并返回我们的神经网络模型的作用。
    # 它需要传入调用 fit()所需要的参数,比如迭代次数和批处理大小。
    # 最新接口指定训练的次数为epochs
    clf = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=128, verbose=0)

    #使用5折交叉验证
    scores = cross_val_score(clf, x, encoded_y, cv=5, scoring='f1_micro')
    # print scores
    print("f1_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #scores = cross_val_score(clf, x, encoded_y, cv=5, scoring='accuracy')
    # print scores
    #print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#使用keras的cnn+mlp
def do_keras_cnn_mlp(text,stars):

    #转换成词袋序列
    max_document_length=200

    #设置分词最大个数 即词袋的单词个数
    tokenizer = Tokenizer(num_words=max_features,lower=True)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    x=pad_sequences(sequences, maxlen=max_document_length)


    #我们可以使用从scikit-learn LabelEncoder类。
    # 这个类通过 fit() 函数获取整个数据集模型所需的编码,然后使用transform()函数应用编码来创建一个新的输出变量。
    encoder=LabelEncoder()
    encoder.fit(stars)
    encoded_y = encoder.transform(stars)



    #构造神经网络
    def baseline_model():

        #CNN参数
        embedding_dims = 50
        filters = 250
        kernel_size = 3
        hidden_dims = 250

        model = Sequential()
        model.add(Embedding(max_features, embedding_dims))

        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        #池化
        model.add(GlobalMaxPooling1D())


        #增加一个隐藏层
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        #输出层

        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        #可视化
        plot_model(model, to_file='yelp-cnn-model-mlp.png',show_shapes=True)

        model.summary()

        return model
    #在 scikit-learn 中使用 Keras 的模型,我们必须使用 KerasClassifier 进行包装。这个类起到创建并返回我们的神经网络模型的作用。
    # 它需要传入调用 fit()所需要的参数,比如迭代次数和批处理大小。
    # 最新接口指定训练的次数为epochs
    clf = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=128, verbose=0)

    #使用5折交叉验证
    scores = cross_val_score(clf, x, encoded_y, cv=5, scoring='f1_micro')
    # print scores
    print("f1_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#使用keras的TextCNN

def do_keras_textcnn(text,stars):

    #转换成词袋序列
    max_document_length=200

    #设置分词最大个数 即词袋的单词个数
    tokenizer = Tokenizer(num_words=max_features,lower=True)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    x=pad_sequences(sequences, maxlen=max_document_length)


    #我们可以使用从scikit-learn LabelEncoder类。
    # 这个类通过 fit() 函数获取整个数据集模型所需的编码,然后使用transform()函数应用编码来创建一个新的输出变量。
    encoder=LabelEncoder()
    encoder.fit(stars)
    encoded_y = encoder.transform(stars)


#论文中的参数：
# Convolutional Neural Networks for Sentence Classification
# Hyperparameters and Training
#For all datasets we use: rectified linear units, filter
#windows (h) of 3, 4, 5 with 100 feature maps each,
#dropout rate (p) of 0.5, l2 constraint (s) of 3, and
#mini-batch size of 50. These values were chosen
#via a grid search on the SST-2 dev set.


    #构造神经网络
    def baseline_model():

        #CNN参数
        embedding_dims = 50
        filters = 100

        # Inputs
        input = Input(shape=[max_document_length])

        # Embeddings layers
        x = Embedding(max_features, embedding_dims)(input)

        # conv layers
        convs = []
        for filter_size in [3,4,5]:
            l_conv = Conv1D(filters=filters, kernel_size=filter_size, activation='relu')(x)
            l_pool = MaxPooling1D()(l_conv)
            l_pool = Flatten()(l_pool)
            convs.append(l_pool)

        merge = concatenate(convs, axis=1)

        out = Dropout(0.2)(merge)

        output = Dense(32, activation='relu')(out)

        output = Dense(units=2, activation='softmax')(output)

        #输出层
        model = Model([input], output)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        #可视化
        plot_model(model, to_file='yelp-cnn-model-textcnn.png',show_shapes=True)

        model.summary()

        return model
    #在 scikit-learn 中使用 Keras 的模型,我们必须使用 KerasClassifier 进行包装。这个类起到创建并返回我们的神经网络模型的作用。
    # 它需要传入调用 fit()所需要的参数,比如迭代次数和批处理大小。
    # 最新接口指定训练的次数为epochs
    clf = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=50, verbose=1)

    #使用5折交叉验证
    scores = cross_val_score(clf, x, encoded_y, cv=5, scoring='f1_micro')
    # print scores
    print("f1_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#use pre train word2vec
def do_keras_textcnn_w2v(text,stars,trainable):

    #转换成词袋序列
    max_document_length=200

    embedding_dims = 300


    #获取已经训练好的词向量
    model = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)

    print model['word'].shape


    #设置分词最大个数 即词袋的单词个数
    tokenizer = Tokenizer(num_words=max_features,lower=True)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    x=pad_sequences(sequences, maxlen=max_document_length)


    #我们可以使用从scikit-learn LabelEncoder类。
    # 这个类通过 fit() 函数获取整个数据集模型所需的编码,然后使用transform()函数应用编码来创建一个新的输出变量。
    encoder=LabelEncoder()
    encoder.fit(stars)
    encoded_y = encoder.transform(stars)

    #labels = to_categorical(np.asarray(labels))也可以进行数据处理

    #获取word到对应数字编号的映射关系
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


    #获取词向量的映射矩阵
    embedding_matrix = np.zeros((max_features + 1, embedding_dims))

    for word, i in word_index.items():

        #编号大于max_features的忽略 该字典是按照字典顺序 所以对应的id不一定是顺序的
        if i > max_features:
            continue

        try:
            embedding_matrix[i] = model[word].reshape(embedding_dims)

        except:
            print "%s not found!" % (word)


        #构造神经网络
    def baseline_model():

        #CNN参数

        #filters个数通常与文本长度相当 便于提取特征
        filters = max_document_length

        # Inputs
        input = Input(shape=[max_document_length])

        # 词向量层，本文使用了预训练word2vec词向量，把trainable设为False
        x = Embedding(max_features + 1,
                                    embedding_dims,
                                    weights=[embedding_matrix],
                                    trainable=trainable)(input)



        # conv layers
        convs = []
        for filter_size in [3,4,5]:
            l_conv = Conv1D(filters=filters, kernel_size=filter_size, activation='relu')(x)
            l_pool = MaxPooling1D()(l_conv)
            l_pool = Flatten()(l_pool)
            convs.append(l_pool)

        merge = concatenate(convs, axis=1)

        out = Dropout(0.2)(merge)

        output = Dense(32, activation='relu')(out)

        output = Dense(units=2, activation='softmax')(output)

        #输出层
        model = Model([input], output)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        #可视化
        plot_model(model, to_file='yelp-cnn-model-textcnn.png',show_shapes=True)

        model.summary()

        return model
    #在 scikit-learn 中使用 Keras 的模型,我们必须使用 KerasClassifier 进行包装。这个类起到创建并返回我们的神经网络模型的作用。
    # 它需要传入调用 fit()所需要的参数,比如迭代次数和批处理大小。
    # 最新接口指定训练的次数为epochs
    clf = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=50, verbose=1)

    #使用5折交叉验证
    scores = cross_val_score(clf, x, encoded_y, cv=5, scoring='f1_micro')
    # print scores
    print("f1_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':


    text,stars=load_reviews(yelp_file,10000)

    stars=[ 0 if star < 3 else 1 for star in stars ]

    print "情感分类的总数:"
    count_classes = pd.value_counts(stars, sort=True)
    print count_classes

    #使用单层cnn文档分类
    #do_keras_cnn(text,stars)

    #使用cnn+mlp文档分类
    #do_keras_cnn_mlp(text,stars)

    #使用textCNN文档分类
    #do_keras_textcnn(text,stars)


    #使用textCNN文档分类 以及预计训练的词向量
    do_keras_textcnn_w2v(text,stars,True)