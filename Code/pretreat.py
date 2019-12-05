"""
Description:定义一些预处理函数
Author：Yang Jiang
Prompt: code in Python3 env
"""

import os
from regular_expression import *

from const import data_dir, common_dir

# 处理分好类的文件利用正则表达式和停用词表进行处理
def pretreat_doc(str_doc):
    str_doc = textParse_news(str_doc)
    str_doc = seg_depart(str_doc)
    return str_doc

#读取停用词列表
def stopwordslist():
    stopword_dir = os.path.join(common_dir, "stopwords.txt")
    stopwords = [line.strip() for line in open(stopword_dir, encoding='UTF-8').readlines()]
    return stopwords

#分词
def seg_depart(sentence):
    #segment for each line
    sentence_depart = jieba.cut(sentence.strip())
    stopwords = stopwordslist()
    outstr = ""
    for word in sentence_depart:
        if word not in stopwords:
            if word != ('/t' and "##"):
                outstr += word
                outstr += " "
    return outstr
