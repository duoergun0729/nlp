# -*- coding: UTF-8 -*-


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


import jieba
import jieba.posseg


def do_posseq(text):
    #seg_lig=jieba.cut(text,cut_all=False)

    seg_lig = jieba.posseg.cut(text)

    for w,tag in seg_lig:
        print "%s /%s" % (w,tag)

if __name__ == '__main__':

    do_posseq("我爱北京天安门")

