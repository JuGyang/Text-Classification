
"""
Description:对分类器进行测试
Author：Yang Jiang
Prompt: code in Python3 env
"""

import os
import time
import logging
import pickle as pkl
import jieba

from const import model_dir
from gensim import corpora,models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from scipy.sparse import csr_matrix

if __name__ == '__main__':

    path_tmp = model_dir + '/CSCMNews_model'
    path_dictionary = os.path.join(path_tmp, 'CSCMNews.dict')
    path_tmp_lsi_model = os.path.join(path_tmp, 'model.lsi.pkl')
    path_tmp_predictor = os.path.join(path_tmp, 'predictor.pkl')
    path_tmp_lsi = os.path.join(path_tmp, 'lsi_corpus')


    dictionary = corpora.Dictionary.load(path_dictionary)

    lsi_file = open(path_tmp_lsi_model,'rb')
    lsi_model = pkl.load(lsi_file)
    lsi_file.close()

    x = open(path_tmp_predictor,'rb')
    predictor = pkl.load(x)
    x.close()

    files = os.listdir(path_tmp_lsi)
    catg_list = []
    for file in files:
        t = file.split('.')[0]
        if t not in catg_list:
            catg_list.append(t)
    demo_doc = """这次大选让两党的精英都摸不着头脑。以媒体专家的传统观点来看，要选总统首先要避免失言，避免说出一些“offensive”的话。希拉里，罗姆尼，都是按这个方法操作的。罗姆尼上次的47%言论是在一个私人场合被偷录下来的，不是他有意公开发表的。今年希拉里更是从来没有召开过新闻发布会。
川普这种肆无忌惮的发言方式，在传统观点看来等于自杀。"""
    print("原文本内容为：")
    print(demo_doc)
    demo_doc = list(jieba.cut(demo_doc,cut_all=False))
    demo_bow = dictionary.doc2bow(demo_doc)
    tfidf_model = models.TfidfModel(dictionary=dictionary)
    demo_tfidf = tfidf_model[demo_bow]
    demo_lsi = lsi_model[demo_tfidf]
    data = []
    cols = []
    rows = []
    for item in demo_lsi:
        data.append(item[1])
        cols.append(item[0])
        rows.append(0)
    demo_matrix = csr_matrix((data,(rows,cols))).toarray()
    x = predictor.predict(demo_matrix)
    print('分类结果为：{x}'.format(x=catg_list[x[0]]))
