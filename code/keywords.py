# -*- coding: UTF-8 -*-

from jieba import analyse
tfidf = analyse.extract_tags

if __name__ == '__main__':

    text="""
    据半岛电视台援引叙利亚国家电视台称，叙利亚已经对美国、英国、法国的空袭进行了反击。据介绍，在叙军武器库中，对西方最具威慑力的当属各型战术地对地弹道导弹。
    尽管美英法是利用巡航导弹等武器发动远程空袭，但叙军要对等还击却几乎是“不可能完成的任务”。目前叙军仍能作战的战机仍是老旧的苏制米格-29、米格-23、米格-21战斗机和苏-22、苏-24轰炸机，它们在现代化的西方空军面前难有自保之力，因此叙军的远程反击只能依靠另一个撒手锏——地对地战术弹道导弹。
    """

    # 关键词提取所使用停用词文本语料库可以切换成自定义语料库的路径。
    analyse.set_stop_words("stopwords.txt")

    # 引入TextRank关键词抽取接口
    textrank = analyse.textrank

    # 基于TextRank算法进行关键词抽取
    keywords_textrank = textrank(text,topK = 10, withWeight = False, allowPOS = ('n','ns','vn','v','nz'))
    # 输出抽取出的关键词
    for keyword in keywords_textrank:
        print keyword + "/"


    print "TFIDF"

    # TFIDF
    keywords_tfidf = analyse.extract_tags(text,topK = 10, withWeight = False, allowPOS = ('n','ns','vn','v','nz'))

    # 输出抽取出的关键词
    for keyword in keywords_tfidf:
        print keyword + "/"