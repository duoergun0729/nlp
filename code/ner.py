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

    #for w,tag in seg_lig:
    #    print "%s /%s" % (w,tag)

    print " ".join(["%s /%s" % (w,tag) for w,tag in seg_lig])

if __name__ == '__main__':

    #do_posseq("我爱北京天安门")

    text = """
        据半岛电视台援引叙利亚国家电视台称，叙利亚已经对美国、英国、法国的空袭进行了反击。据介绍，在叙军武器库中，对西方最具威慑力的当属各型战术地对地弹道导弹。
        尽管美英法是利用巡航导弹等武器发动远程空袭，但叙军要对等还击却几乎是“不可能完成的任务”。目前叙军仍能作战的战机仍是老旧的苏制米格-29、米格-23、米格-21战斗机和苏-22、苏-24轰炸机，它们在现代化的西方空军面前难有自保之力，因此叙军的远程反击只能依靠另一个撒手锏——地对地战术弹道导弹。
        """

    do_posseq(text)

