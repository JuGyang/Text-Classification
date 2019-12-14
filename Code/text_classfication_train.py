# coding:utf8

"""
Description:gensim实现新闻文本特征向量化,并训练分类器
Author：Yang Jiang
Prompt: code in Python3 env
"""

import os
import time
import sys
import logging
import pickle as pkl
import numpy as np
import json

from pretreat import *
from const import data_dir, model_dir, cache_dir
from gensim import corpora, models
from sklearn import svm
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib



#读取文件
class loadFolders(object):   # 迭代器
    def __init__(self, par_path):
        self.par_path = par_path
    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath): # if file is a folder
                yield file_abspath

class loadFiles(object):
    def __init__(self, par_path):
        self.par_path = par_path
    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:              # level directory
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder):     # secondary directory
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    this_file = open(file_path, 'rb') #rb读取方式更快
                    content = this_file.read().decode('utf8', 'ignore')
                    yield catg, content
                    this_file.close()

def svm_classify(train_set,train_tag,test_set,test_tag):

    clf = svm.SVC(kernel='rbf', C=4, gamma='scale')
    clf_res = clf.fit(train_set,train_tag)
    #train_pred  = clf_res.predict(train_set)
    print('{t} === 分类器生成完毕，开始分类 ==='.format(t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
    test_pred = clf_res.predict(test_set)
    files = os.listdir(path_tmp_lsi)
    catg_list = []
    for file in files:
        t = file.split('.')[0]
        if t not in catg_list:
            catg_list.append(t)

    print('{t} === 分类训练完毕，分类结果如下 ==='.format(t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
    print(classification_report(test_tag, test_pred, target_names=catg_list))

    return clf_res


if __name__=='__main__':
    start = time.time()

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )

    n = 1  # n 表示抽样率
    path_doc_root = data_dir + '/News'  # 根目录 即存放按类分类好的文本数据集
    path_tmp = model_dir + '/News_model'  # 存放中间结果的位置
    path_dictionary = os.path.join(path_tmp, 'News.dict')
    path_news = os.path.join(cache_dir, 'news.txt')
    path_tmp_tfidf = os.path.join(path_tmp, 'tfidf_corpus')
    path_tmp_lsi = os.path.join(path_tmp, 'lsi_corpus')
    path_tmp_lsi_model = os.path.join(path_tmp, 'model.lsi.pkl')
    path_tmp_predictor = os.path.join(path_tmp, 'svm.model')
    path_vec_corpus = os.path.join(path_tmp, 'news_corpus.mm')
    path_test_set = os.path.join(cache_dir, 'test_set.pkl')
    path_test_tag = os.path.join(cache_dir, 'test_tag.pkl')

    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    #清空缓存
    dictionary = None
    corpus_tfidf = None
    corpus_lsi = None
    lsi_model = None
    predictor = None

    news = [] #用于存储新闻
    corpus = [] #用于存储语料

    # # ===================================================================
    # #  第一阶段，遍历文档，生成词典,并去掉频率较少的项
    #       如果指定的位置没有词典，则重新生成一个。如果有，则跳过该阶段
    if not os.path.exists(path_dictionary):
        print('=== 未检测到有词典存在，开始遍历生成词典 ===')
        dictionary = corpora.Dictionary()
        files = loadFiles(path_doc_root)
        for i, msg in enumerate(files):
            if i % n == 0:
                catg = msg[0]
                content = msg[1]
                #corpus用来记录所有的新闻语料的list
                news.append(textParse_corpus(content)) #textParse是正则表达式函数
                content = seg_doc(content)
                dictionary.add_documents([content])
                if int(i/n) % 1000 == 0:
                    print('{t} *** {i} \t docs has been dealed'
                          .format(i=i, t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        # 去掉词典中出现次数过少的
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5 ]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify() # 重新产生连续的编号
        dictionary.save(path_dictionary)
        print('===news的长度===')
        print(len(news))
        print('=== 词典已经生成 ===')
        if not os.path.exists('path_news'):
            print('=== 未检测到有新闻语料存在，开始遍历生成新闻语料 ===')
            file_news = open(path_news, 'w', encoding='UTF-8')
            for new in news:
                file_news.write(new)
                file_news.write('\n')
            file_news.close()
            print('=== 新闻语料已经生成 ===')
        else:
            print('=== 检测到新闻语料已经存在，跳过该阶段 ===')

    else:
        print('=== 检测到词典已经存在，跳过该阶段 ===')

    # # ===================================================================
    # # # 第二阶段，  开始将文档转化成tfidf
    dictionary = None
    if not os.path.exists(path_tmp_tfidf):
        print('=== 未检测到有tfidf文件夹存在，开始生成tfidf向量 ===')
        # 如果指定的位置没有tfidf文档，则生成一个。如果有，则跳过该阶段
        if not dictionary:  # 如果跳过了第一阶段，则从指定位置读取词典
            dictionary = corpora.Dictionary.load(path_dictionary)
        os.makedirs(path_tmp_tfidf)
        files = loadFiles(path_doc_root)
        tfidf_model = models.TfidfModel(dictionary=dictionary)
        corpus_tfidf = {}
        for i, msg in enumerate(files):
            if i % n == 0:
                catg = msg[0]
                content = msg[1]
                word_list = seg_doc(content)
                file_bow = dictionary.doc2bow(word_list)
                corpus.append(file_bow)
                file_tfidf = tfidf_model[file_bow]
                tmp = corpus_tfidf.get(catg, [])
                tmp.append(file_tfidf)
                if tmp.__len__() == 1:
                    corpus_tfidf[catg] = tmp
            if i % 10000 == 0:
                print('{i} files is dealed'.format(i=i))
        corpora.MmCorpus.serialize(path_vec_corpus, corpus)

        # 将tfidf中间结果储存起来
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_tfidf, s=os.sep, c=catg),corpus_tfidf.get(catg),id2word=dictionary)
            print('catg {c} has been transformed into tfidf vector'.format(c=catg))
        print('=== tfidf向量已经生成 ===')
    else:
        print('=== 检测到tfidf向量已经生成，跳过该阶段 ===')
    if not os.path.exists(path_tmp_lsi):
        print('=== 未检测到有LSI文件夹存在，开始生成LSI模型 ===')

        if not dictionary:
            dictionary = corpora.Dictionary.load(path_dictionary)
        if not corpus_tfidf:

            print('=== 未检测到tfidf文档，开始从磁盘中读取 ===')
            # 从对应文件夹中读取所有类别
            files = os.listdir(path_tmp_tfidf)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)

            # 从磁盘中读取corpus
            corpus_tfidf = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_tfidf,s=os.sep,c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_tfidf[catg] = corpus
            print('=== tfidf文档读取完毕，开始转化成lsi向量 ===')

        #开始生成lsi model
        os.makedirs(path_tmp_lsi)
        corpus_tfidf_total = []
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            tmp = corpus_tfidf.get(catg)
            corpus_tfidf_total += tmp
        lsi_model = models.LsiModel(corpus = corpus_tfidf_total, id2word=dictionary, num_topics=150, chunksize=20000)
        #将lsi模型存储到磁盘上
        lsi_file = open(path_tmp_lsi_model, 'wb')
        pkl.dump(lsi_model, lsi_file)
        lsi_file.close()
        del corpus_tfidf_total

         # 生成corpus of lsi, 并逐步去掉 corpus of tfidf
        corpus_lsi = {}
        for catg in catgs:
            corpus = [lsi_model[doc] for doc in corpus_tfidf.get(catg)]
            corpus_lsi[catg] = corpus
            corpus_tfidf.pop(catg)
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_lsi,s=os.sep,c=catg),
                                       corpus,
                                       id2word=dictionary)

        print('=== LSI向量生成完毕 ===')

    else:
        print('=== 检测到LSI向量已经生成，跳过该阶段 ===')
    if not os.path.exists(path_tmp_predictor):
        print('=== 未检测到分类器存在,开始进行分类过程 ===')
        if not corpus_lsi: # 如果跳过了第三阶段
            print('=== 未检测到lsi文档，开始从磁盘中读取 ===')
            files = os.listdir(path_tmp_lsi)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)
            # 从磁盘中读取corpus
            corpus_lsi = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_lsi,s=os.sep,c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_lsi[catg] = corpus
            print('=== lsi文档读取完毕，开始进行分类 ===')
        tag_list = []
        doc_num_list = []
        corpus_lsi_total = []
        catg_list = []
        files = os.listdir(path_tmp_lsi)
        for file in files:
            t = file.split('.')[0]
            if t not in catg_list:
                catg_list.append(t)
        for count,catg in enumerate(catg_list):
            tmp = corpus_lsi[catg]
            tag_list += [count]*tmp.__len__()
            doc_num_list.append(tmp.__len__())
            corpus_lsi_total += tmp
            corpus_lsi.pop(catg)

        # 将gensim中的mm表示转化成numpy矩阵表示
        data = []
        rows = []
        cols = []
        line_count = 0
        for line in corpus_lsi_total:
            for elem in line:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1
        lsi_matrix = csr_matrix((data,(rows,cols))).toarray()
        # 生成训练集和测试集
        rarray=np.random.random(size=line_count)
        train_set = []
        train_tag = []
        test_set = []
        test_tag = []
        for i in range(line_count):
            if rarray[i]<0.5:
                train_set.append(lsi_matrix[i,:])
                train_tag.append(tag_list[i])
            else:
                test_set.append(lsi_matrix[i,:])
                test_tag.append(tag_list[i])

        test_set_pkl = open(path_test_set, 'wb')
        pkl.dump(test_set, test_set_pkl)
        test_set_pkl.close()
        test_tag_pkl = open(path_test_tag, 'wb')
        pkl.dump(test_tag, test_tag_pkl)
        test_tag_pkl.close()
        print('=== 训练集 测试集 生成完毕 ===')
        # 生成分类器
        predictor = svm_classify(train_set,train_tag,test_set,test_tag)
        joblib.dump(predictor, path_tmp_predictor)
    else:
        print('=== 检测到分类器已经生成，跳过该阶段 ===')
        predictor = joblib.load(path_tmp_predictor)
        print('{t} === 分类器加载完毕 ==='.format(t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        test_set_pkl = open(path_test_set, 'rb')
        test_set = pkl.load(test_set_pkl)
        test_set_pkl.close()
        print('{t} === 测试集加载完毕 ==='.format(t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        test_tag_pkl = open(path_test_tag, 'rb')
        test_tag = pkl.load(test_tag_pkl)
        test_tag_pkl.close()
        print('{t}=== 测试标签加载完毕 ==='.format(t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        test_pred = predictor.predict(test_set)
        files = os.listdir(path_tmp_lsi)
        catg_list = []
        for file in files:
            t = file.split('.')[0]
            if t not in catg_list:
                catg_list.append(t)

        print('{t}=== 分类完毕，分类结果如下 ==='.format(t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        print(classification_report(test_tag, test_pred, target_names=catg_list))


    end = time.time()
    print('total spent times:%.2f' % (end-start)+ ' s')
