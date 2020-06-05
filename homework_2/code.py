import json
import re
from copy import deepcopy
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from functools import cmp_to_key

def cmp_paper(p1, p2):
    try:
        y1 = int(p1['year']) if 'year' in p1 else 0
    except:
        y1 = 9999
    if y1 < 1000 or y1 > 2100:
        y1 = 9999
    try:
        y2 = int(p2['year']) if 'year' in p2 else 0
    except:
        y2 = 9999
    if y2 < 1000 or y2 > 2100:
        y2 = 9999
    if y1 < y2:
        return -1
    elif y1 > y2:
        return 1
    return 0

  class DataProcessing:

    def __init__(self):
        # key = 0: train data that has cluster; key = 1: valid data that not have cluster.
        self.key = 1
        row_data_path_train = '/root/lyz/tsinghua/data/train/train_author.json'
        pub_data_path_train = '/root/lyz/tsinghua/data/train/train_pub.json'
        row_data_path_valid = '/root/lyz/tsinghua/data/sna_data/sna_valid_author_raw.json'
        pub_data_path_valid = '/root/lyz/tsinghua/data/sna_data/sna_valid_pub.json'
        # row_data_path_valid = './data/sna_test_data/sna_test_author_raw.json'
        # pub_data_path_valid = './data/sna_test_data/test_pub_sna.json'

        if self.key == 0:
            pub_data = json.load(open(pub_data_path_train, 'r', encoding='utf-8'))
            self.data = json.load(open(row_data_path_train, 'r', encoding='utf-8'))
        else:
            pub_data = json.load(open(pub_data_path_valid, 'r', encoding='utf-8'))
            self.data = json.load(open(row_data_path_valid, 'r', encoding='utf-8'))
        self.raw_data = deepcopy(self.data)
        self.authorList = []
        if self.key == 0:
            for author in self.data:
                author_data = []
                for author_id in self.data[author]:
                    author_data += [pub_data[paper_id] for paper_id in self.data[author][author_id]]
                self.data[author] = author_data
                self.authorList.append(author)
        else:
            for author in self.data:
                self.data[author] = [pub_data[paper_id] for paper_id in self.data[author]]
                self.authorList.append(author)

        # mapping_orgs
        self.past_text = []
        self.past_id = []
        self.now_id = 0

        # uni_replace
        self.uni = \
            "\u00e0 \u00e1 \u00e2 \u00e3 \u00e4 \u00e5 \u00e6 \u00e7 \u00e8 \u00e9 \u00ea \u00eb \u00ec \u00ed \u00ef "\
            "\u00f1 \u00f2 \u00f3 \u00f4 \u00f6 \u00f8 \u00f9 \u00fa \u00fc \u00fd \u0103 \u0107 \u010d \u011b"
        self.uni_replace = \
            "a a a a a a a c e e e e i i i " \
            "n o o o o o u u u y a c c e"

    def precessname(self, name):
        name = name.lower().replace(' ', '_')
        name = name.replace('.', '_')
        name = name.replace('-', '')
        name = re.sub(r"_{2,}", "_", name)
        name = name.split('_')
        name.sort()
        name = '_'.join(name)
        return name

    # 预处理机构,简写替换，
    def preprocessorg(self, org):
        if org != "":
            org = org.replace('Sch.', 'School')
            org = org.replace('Dept.', 'Department')
            org = org.replace('Coll.', 'College')
            org = org.replace('Inst.', 'Institute')
            org = org.replace('Univ.', 'University')
            org = org.replace('Lab ', 'Laboratory ')
            org = org.replace('Lab.', 'Laboratory')
            org = org.replace('Natl.', 'National')
            org = org.replace('Comp.', 'Computer')
            org = org.replace('Sci.', 'Science')
            org = org.replace('Tech.', 'Technology')
            org = org.replace('Technol.', 'Technology')
            org = org.replace('Elec.', 'Electronic')
            org = org.replace('Engr.', 'Engineering')
            org = org.replace('Aca.', 'Academy')
            org = org.replace('Syst.', 'Systems')
            org = org.replace('Eng.', 'Engineering')
            org = org.replace('Res.', 'Research')
            org = org.replace('Appl.', 'Applied')
            org = org.replace('Chem.', 'Chemistry')
            org = org.replace('Prep.', 'Petrochemical')
            org = org.replace('Phys.', 'Physics')
            org = org.replace('Phys.', 'Physics')
            org = org.replace('Mech.', 'Mechanics')
            org = org.replace('Mat.', 'Material')
            org = org.replace('Cent.', 'Center')
            org = org.replace('Ctr.', 'Center')
            org = org.replace('Behav.', 'Behavior')
            org = org.replace('Atom.', 'Atomic')
            org = ' '.join(org.split(';'))
        return org

    def de_stopwords(self, key, content):
        stemmer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        if key == 0:
            stop_words.add('university')
            stop_words.add('institute')
            stop_words.add('department')
        if key == 1:
            stop_words.add('study')
            stop_words.add('based')
            stop_words.add('effect')
            stop_words.add('analysis')
            stop_words.add('system')
            stop_words.add('method')
            stop_words.add('using')
            stop_words.add('model')
            stop_words.add('synthesis')
            stop_words.add('application')
        if key == 2:
            stop_words.add('result')
            stop_words.add('method')
            stop_words.add('using')
            stop_words.add('based')
            stop_words.add('high')
            stop_words.add('used')
            stop_words.add('different')
            stop_words.add('show')
            stop_words.add('also')
            stop_words.add('two')
            stop_words.add('proposed')
            stop_words.add('structure')
            stop_words.add('al')
            stop_words.add('et')
            stop_words.add('performance')
            stop_words.add('new')
            stop_words.add('respectively')
            stop_words.add('compared')
            stop_words.add('higher')
            stop_words.add('one')
            stop_words.add('could')
            stop_words.add('however')
            stop_words.add('may')
            stop_words.add('found')
            stop_words.add('obtained')
            stop_words.add('significant')
            stop_words.add('case')
            stop_words.add('three')
            stop_words.add('well')
            stop_words.add('important')
            stop_words.add('design')
            stop_words.add('type')
            stop_words.add('conclusion')
        if key == 3:
            stop_words.add('journal')
            stop_words.add('university')
            stop_words.add('conference')

        # 词干提取
        content = [stemmer.lemmatize(word) for word in content.split()
                   if word not in stop_words and len(word) > 1]
        return ' '.join(content)

    # 正则去标点
    def etl(self, content):
        content = re.sub("[\s+\.\!\/,;|:${&~}<=>?\.%^\[\]*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ", content)
        # 不去 '-'
        # content = re.sub(r" \d+\b", " ", content)  # 去数字
        # content = re.sub(r"\b\d+ ", " ", content)  # 去数字
        # 即保留 -3，3-这种形式的数字
        content = re.sub(r" {2,}", " ", content)
        return content

    def generateDataset(self):
        i = 0
        for authorNo in range(len(self.authorList)):
            # if authorNo > 0:
            #     continue
            print('generating author json No.' + str(authorNo) + ' !!!')
            author = self.authorList[authorNo]
            orgs_author = self.precessname(author)
            # 此处进行记录paper_id2cluster原因：不同的paper会属于不同簇，共同作者导致的。
            paper_id2cluster = {}
            if self.key == 0:
                for author_id in self.raw_data[author]:
                    for paper_id in self.raw_data[author][author_id]:
                        paper_id2cluster[paper_id] = i
                    i += 1
            res_dict = {'paperID': '', 'clusterNo': -1}
            papers = self.data[author]
            if len(papers) == 0:
                self.authorList[authorNo] = 'NoPaper'
                continue
            with open('data/WhoIsWho-' + str(authorNo), 'w'):
                pass
            self.past_text.clear()
            self.past_id.clear()
            self.now_id = 0

            # 按发表年份排序
            papers = sorted(papers, key=cmp_to_key(cmp_paper))

            for paper in papers:
                authors = paper['authors']

                names = []
                orgs = []
                for _1, paper_author in enumerate(authors):
                    name = paper_author['name']
                    for _2 in range(len(self.uni)):
                        name = name.replace(self.uni[_2], self.uni_replace[_2])
                    name = self.precessname(name)
                    if 'org' in paper_author:
                        # if paper_author['org'] != '':
                        orgs.append(self.de_stopwords(0, self.etl(self.preprocessorg(paper_author['org']).lower())))
                    else:
                        orgs.append('')
                    # names.append(name)
                    if name == orgs_author:
                        if orgs[-1] != '':
                            names.append(orgs[-1])
                    else:
                        names.append(name)

                title = paper['title']
                abstract = paper['abstract'] if 'abstract' in paper else ''
                if abstract is None:
                    abstract = ''
                keywords = paper['keywords'] if 'keywords' in paper else []
                venue = paper["venue"] if 'venue' in paper else ''

                # ------------------------------------------------------------------------
                res_dict['paperID'] = paper['id']
                if self.key == 0:
                    res_dict['clusterNo'] = paper_id2cluster[paper['id']]

                res_dict['co-authors'] = self.etl(' '.join(names))

                res_dict['orgs'] = ' '.join(orgs)
                res_dict['title'] = self.de_stopwords(1, self.etl(title.lower()))
                res_dict['abstract'] = self.de_stopwords(2, self.etl(abstract.lower()))
                res_dict['keywords'] = self.etl(' '.join(keywords).lower())
                res_dict['venue'] = self.de_stopwords(3, self.etl(venue.lower()))
                # ------------------------------------------------------------------------

                with open('data/WhoIsWho-' + str(authorNo), 'a+') as f:
                    f.write(json.dumps(res_dict) + '\n')

        print('generating author authorList !!!')
        with open('data/WhoIsWho-authorList', 'w') as f:
            f.write(' '.join(self.authorList))

class Document:

    # python中传递列表等对象变量是引用传递
    def __init__(self, text, wordToIdMap, wordList, documentID):    # 文档文本，(all)word到id的映射字典，(all)word列表
        self.documentID = documentID    # 文档id
        self.wordIdArray = [[] for _ in range(len(text))]   # 文档中出现的word的id列表
        self.wordFreArray = [[] for _ in range(len(text))]  # 文档中出现的每个word的频率列表
        self.wordNum = []   # 文档中出现的word个数
        self.wordToIdMap = wordToIdMap
        self.wordList = wordList
        for idx, text_idx in enumerate(text):
            self.parseText(text_idx, idx)

    def parseText(self, text_idx, idx):
        V = len(self.wordToIdMap)
        wordFreMap = {}
        ws = text_idx.strip().split(' ')
        for w in ws:
            if w not in self.wordToIdMap:
                V += 1
                wId = V
                self. wordToIdMap[w] = V
                self.wordList.append(w)
            else:
                wId = self.wordToIdMap[w]

            if wId not in wordFreMap:
                wordFreMap[wId] = 1
            else:
                wordFreMap[wId] = wordFreMap[wId] + 1
                # wordFreMap[wId] = 1     # 始终记录频率 = 1
        self.wordNum.append(wordFreMap.__len__())
        w = 0
        for wfm in wordFreMap:  # 等价于wordFreMap.keys()
            self.wordIdArray[idx].append(wfm)
            self.wordFreArray[idx].append(wordFreMap[wfm])
            w += 1

            
import json


class DocumentSet:

    def __init__(self, dataDir, wordToIdMap, wordList):
        self.D = 0  # The number of documents
        self.Lambda = [0.8, 0.1, 0.1]
        # self.clusterNoArray = []
        self.documents = []     # document对象列表，不含文本数据，所以占内存并不太大
        with open(dataDir) as f:
            line = f.readline()
            while line:
                self.D += 1
                obj = json.loads(line)
                text1 = obj['co-authors'] + (' ' + obj['orgs'] if obj['orgs'] != '' else '')
                text2 = obj['title'] + (' ' + obj['abstract'] if obj['abstract'] != '' else '')
                text3 = obj['keywords']
                text = [text1, text2, text3]
                document = Document(text, wordToIdMap, wordList, obj['paperID'])
                self.documents.append(document)
                line = f.readline()
        print("number of documents is ", self.D)
        

import random
import os
import sys
import math
import copy


class Model:

    def __init__(self, K, All_K, V, iterNum, alpha0, beta, threshold_init, threshold_fix, dataset, ParametersStr,
                 sampleNo, wordsInTopicNum):
        self.K = K
        self.All_K = All_K
        self.V = V
        self.iterNum = iterNum
        self.alpha0 = alpha0
        self.beta = beta
        self.threshold_init = threshold_init
        self.threshold_fix = threshold_fix
        self.dataset = dataset
        self.ParametersStr = ParametersStr
        self.sampleNo = sampleNo
        self.wordsInTopicNum = copy.deepcopy(wordsInTopicNum)

        self.Kin = K
        self.beta0 = [float(V) * float(beta)]
        self.phi_zv = []

        self.smallDouble = 1e-150
        self.largeDouble = 1e150

    def run_GSDPMM(self, documentSet, outputPath, wordList):
        self.D_All = documentSet.D  # The whole number of documents
        self.Lambda = documentSet.Lambda
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        # The number of words in cluster z                   (clusterID -> number of words)
        self.n_z = [{} for _ in range(len(self.Lambda))]
        # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.n_zv = [{} for _ in range(len(self.Lambda))]
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        # deepcopy 从原object中真正独立出来，即改变K_current而不会影响K（对于list等对象会有影响）。
        self.K_current = copy.deepcopy(self.K)  # the number of cluster containing documents currently
        # self.BatchSet = {} # No need to store information of each batch
        self.word_current = [[] for _ in range(len(self.Lambda))]  # Store word-IDs' list of Database

        while self.currentDoc < self.D_All:
            self.hasIntialized = {}  # represents the situation of each document
            self.K_last = self.K  # record the last K of the cluster generated before this batch
            self.intialize(documentSet)
            self.gibbsSampling(documentSet)
            self.hasIntialized.clear()
            print("\tGibbs sampling successful! Start to saving results.")
            self.output(documentSet, outputPath, wordList)
            print("\tSaving successful!")
            # print("\tGibbs sampling successful! Saving to res_dict.")
        return self.z, self.All_K + self.K + 1

    # Get beta0 for every batch
    def getBeta0(self):
        res = []
        for e in self.word_current:
            Words = e
            res.append(float(len(list(set(Words)))) * float(self.beta))
        return res

    def intialize(self, documentSet):
        self.word_current = [[] for _ in range(len(self.Lambda))]
        self.D = self.D_All

        # beta取决于当前单词类别数
        self.beta0 = self.getBeta0()
        # alpha取决于整个文档数
        self.alpha = self.alpha0 * self.D
        print("\t" + str(self.D) + " documents will be analyze. alpha is" + " %.2f." % self.alpha +
              " beta is" + " %f." % self.beta +
              "\n\tInitialization.", end='\t')
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID
            # This method is getting beta0 before each document is initialized
            for idx in range(len(self.Lambda)):
                for w in range(document.wordNum[idx]):
                    wordNo = document.wordIdArray[idx][w]
                    if wordNo not in self.word_current[idx]:
                        self.word_current[idx].append(wordNo)
            # beta取决于当前前四个batch单词类别数 + 当前已经初始化过的文档中单词类别数
            self.beta0 = self.getBeta0()
            for beta in self.beta0:
                if beta <= 0:
                    print("Wrong V!")
                    exit(-1)

            if documentID not in self.hasIntialized:
                self.hasIntialized[documentID] = 0  # 0 means that it wasn't initialized
            # iterNum对于一个文档重复迭代的次数
            cluster = None
            if self.iterNum != 0:
                cluster = self.sampleCluster(d, document, documentID, "init")
            if self.iterNum == 0:
                cluster = self.sampleCluster(d, document, documentID, "first")
            # 1 means that documents have been initialized but wasn't fixed
            # 2 means that documents have been fixed
            if self.hasIntialized[documentID] == 1 or self.hasIntialized[documentID] == 2:
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for idx in range(len(self.Lambda)):
                    for w in range(document.wordNum[idx]):
                        wordNo = document.wordIdArray[idx][w]
                        wordFre = document.wordFreArray[idx][w]
                        if cluster not in self.n_zv[idx]:
                            self.n_zv[idx][cluster] = {}
                        if wordNo not in self.n_zv[idx][cluster]:
                            self.n_zv[idx][cluster][wordNo] = 0
                        self.n_zv[idx][cluster][wordNo] += wordFre
                        if cluster not in self.n_z[idx]:
                            self.n_z[idx][cluster] = 0
                        self.n_z[idx][cluster] += wordFre
            if d == self.D_All - 1:
                self.startDoc = self.currentDoc
                self.currentDoc = self.D_All

        # Initialize documents which weren't initialized. 即文档没被信任地归入cluster，即小于threshold_init
        notInitialized = 0
        for d in range(self.startDoc, self.currentDoc):
            document = documentSet.documents[d]
            documentID = document.documentID
            if self.hasIntialized[documentID] == 1 or self.hasIntialized[documentID] == 2:
                continue
            if self.hasIntialized[documentID] == 0:
                notInitialized += 1
                cluster = self.sampleCluster(d, document, documentID, "first")
                if self.hasIntialized[documentID] == 1 or self.hasIntialized[documentID] == 2:
                    self.z[documentID] = cluster
                    if cluster not in self.m_z:
                        self.m_z[cluster] = 0
                    self.m_z[cluster] += 1
                    for idx in range(len(self.Lambda)):
                        for w in range(document.wordNum[idx]):
                            wordNo = document.wordIdArray[idx][w]
                            wordFre = document.wordFreArray[idx][w]
                            if cluster not in self.n_zv[idx]:
                                self.n_zv[idx][cluster] = {}
                            if wordNo not in self.n_zv[idx][cluster]:
                                self.n_zv[idx][cluster][wordNo] = 0
                            self.n_zv[idx][cluster][wordNo] += wordFre
                            if cluster not in self.n_z[idx]:
                                self.n_z[idx][cluster] = 0
                            self.n_z[idx][cluster] += wordFre
        print("\t", notInitialized, "documents weren't initialized still.")

    def gibbsSampling(self, documentSet):
        for i in range(self.iterNum):
            print("\titer is ", i + 1, end="\t")
            # self.updateBeta()  # update beta
            print("beta is" + " %f." % self.beta, end='\t')
            notInitialized = 0  # number of documents not initialized
            notfixed = 0  # number of documents initialized but not fixed
            hasfixed = 0  # number of documents have been fixed
            for d in range(self.startDoc, self.currentDoc):
                document = documentSet.documents[d]
                documentID = document.documentID
                if self.hasIntialized[documentID] == 2:  # documents fixed
                    hasfixed += 1
                    continue
                if self.hasIntialized[documentID] == 1:
                    notfixed += 1
                    cluster = self.z[documentID]
                    self.m_z[cluster] -= 1
                    for idx in range(len(self.Lambda)):
                        for w in range(document.wordNum[idx]):
                            wordNo = document.wordIdArray[idx][w]
                            wordFre = document.wordFreArray[idx][w]
                            self.n_zv[idx][cluster][wordNo] -= wordFre
                            self.n_z[idx][cluster] -= wordFre
                    self.checkEmpty(cluster)
                    if i == 0:  # if first iteration
                        # cluster = self.sampleCluster(d, document, documentID, "first")
                        cluster = self.sampleCluster(d, document, documentID, "iter")
                    elif i != 0 and i != self.iterNum - 1:  # if not first iteration and not last iteration
                        cluster = self.sampleCluster(d, document, documentID, "iter")
                    elif i == self.iterNum - 1:  # if last iteration
                        cluster = self.sampleCluster(d, document, documentID, "last")
                    self.z[documentID] = cluster
                    if cluster not in self.m_z:
                        self.m_z[cluster] = 0
                    self.m_z[cluster] += 1
                    for idx in range(len(self.Lambda)):
                        for w in range(document.wordNum[idx]):
                            wordNo = document.wordIdArray[idx][w]
                            wordFre = document.wordFreArray[idx][w]
                            if cluster not in self.n_zv[idx]:
                                self.n_zv[idx][cluster] = {}
                            if wordNo not in self.n_zv[idx][cluster]:
                                self.n_zv[idx][cluster][wordNo] = 0
                            if cluster not in self.n_z[idx]:
                                self.n_z[idx][cluster] = 0
                            self.n_zv[idx][cluster][wordNo] += wordFre
                            self.n_z[idx][cluster] += wordFre
            print("\t", notInitialized, "documents weren't initialized still.", notfixed,
                  "documents weren't fixed still.")
            if notfixed == 0:
                print("All document has been initialized! Stop iteration.")
                return

    def sampleCluster(self, d, document, documentID, MODE):
        self.prob = [float(0.0)] * (self.K + 1)
        self.prob_idx = [[float(0.0) for _ in range(self.K + 1)] for _ in range(len(self.Lambda))]
        overflowCount_idx = [[0 for _ in range(self.K + 1)] for _ in range(len(self.Lambda))]

        # keywords等不存在时，将其权重加至co-author
        true_Lambda = self.Lambda
        for idx in range(len(self.Lambda)):
            if document.wordNum[idx] == 0:
                true_Lambda[0] += true_Lambda[idx]
                true_Lambda[idx] = 0.0

        for cluster in range(self.K):
            # 不考虑流数据过时的cluster
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                self.prob[cluster] = 0
                continue

            for idx in range(len(self.Lambda)):
                # self.prob_idx[idx][cluster] = self.m_z[cluster]  # / (self.D - 1 + self.alpha)
                self.prob_idx[idx][cluster] = 1.0  # do not consider completeness

            # 对不同的Lambda分别进行计算
            valueOfRule2_idx = [float(1.0)] * (len(self.Lambda))
            for idx in range(len(self.Lambda)):
                i = 0
                for w in range(document.wordNum[idx]):
                    wordNo = document.wordIdArray[idx][w]
                    wordFre = document.wordFreArray[idx][w]
                    for j in range(wordFre):
                        if wordNo not in self.n_zv[idx][cluster]:
                            self.n_zv[idx][cluster][wordNo] = 0
                        if valueOfRule2_idx[idx] < self.smallDouble:
                            overflowCount_idx[idx][cluster] -= 1
                            valueOfRule2_idx[idx] *= self.largeDouble
                        valueOfRule2_idx[idx] *= (self.n_zv[idx][cluster][wordNo] +
                                                  self.beta + j) / (self.n_z[idx][cluster] + self.beta0[idx] + i)
                        i += 1

            for idx in range(len(self.Lambda)):
                if document.wordNum[idx] == 0:
                    self.prob_idx[idx][cluster] = -1.0
                else:
                    self.prob_idx[idx][cluster] *= valueOfRule2_idx[idx]

        for idx in range(len(self.Lambda)):
            # self.prob_idx[idx][self.K] = self.alpha  # / (self.D - 1 + self.alpha)
            self.prob_idx[idx][self.K] = 1.0    # do not consider completeness

        valueOfRule2_idx = [float(1.0)] * (len(self.Lambda))
        for idx in range(len(self.Lambda)):
            i = 0
            for w in range(document.wordNum[idx]):
                wordFre = document.wordFreArray[idx][w]
                for j in range(wordFre):
                    if valueOfRule2_idx[idx] < self.smallDouble:
                        overflowCount_idx[idx][self.K] -= 1
                        valueOfRule2_idx[idx] *= self.largeDouble
                    valueOfRule2_idx[idx] *= (self.beta + j) / (self.beta0[idx] + i)
                    i += 1

        for idx in range(len(self.Lambda)):
            if document.wordNum[idx] == 0:
                self.prob_idx[idx][self.K] = -1.0
            else:
                self.prob_idx[idx][self.K] *= valueOfRule2_idx[idx]

        self.reComputeProbs(overflowCount_idx, self.K)

        prob_normalized = [float(0.0)] * (self.K + 1)  # record normalized probabilities
        for idx in range(len(self.Lambda)):
            if document.wordNum[idx] == 0:
                continue
            allProb = 0  # record the amount of all probabilities
            for k in range(self.K + 1):
                allProb += self.prob_idx[idx][k]
            for k in range(self.K + 1):
                try:
                    prob_normalized[k] += (self.prob_idx[idx][k] / allProb) * true_Lambda[idx]
                except ZeroDivisionError:
                    print("ZeroDivisionError: float division by zero")
                    print("allProb", allProb)
                    print("K", self.K)
        self.prob = copy.deepcopy(prob_normalized)

        if MODE == "init" or MODE == "1":
            for k in range(self.K + 1):
                if prob_normalized[k] > self.threshold_fix:
                    self.hasIntialized[documentID] = 2
                    break
                if prob_normalized[k] > self.threshold_init:
                    self.hasIntialized[documentID] = 1
                    break
            if self.hasIntialized[documentID] == 0:
                return -1
            kChoosed = 0
            if self.hasIntialized[documentID] == 2:
                bigPro = self.prob[0]
                for k in range(1, self.K + 1):
                    if self.prob[k] > bigPro:
                        bigPro = self.prob[k]
                        kChoosed = k
                if kChoosed == self.K:
                    self.K += 1
                    self.K_current += 1
            elif self.hasIntialized[documentID] == 1:   # 随机sample一个cluster，待细究
                for k in range(1, self.K + 1):
                    self.prob[k] += self.prob[k - 1]
                thred = random.random() * self.prob[self.K]
                while kChoosed < self.K + 1:
                    if thred < self.prob[kChoosed]:
                        break
                    kChoosed += 1
                if kChoosed == self.K:
                    self.K += 1
                    self.K_current += 1
            return kChoosed

        elif MODE == "first" or MODE == "2":
            if self.hasIntialized[documentID] == 0:
                self.hasIntialized[documentID] = 1
            for k in range(self.K + 1):
                if prob_normalized[k] > self.threshold_fix:
                    self.hasIntialized[documentID] = 2
                    break
            kChoosed = 0
            if self.hasIntialized[documentID] == 2:
                bigPro = self.prob[0]
                for k in range(1, self.K + 1):
                    if self.prob[k] > bigPro:
                        bigPro = self.prob[k]
                        kChoosed = k
                if kChoosed == self.K:
                    self.K += 1
                    self.K_current += 1
            elif self.hasIntialized[documentID] == 1:
                for k in range(1, self.K + 1):
                    self.prob[k] += self.prob[k - 1]
                thred = random.random() * self.prob[self.K]
                while kChoosed < self.K + 1:
                    if thred < self.prob[kChoosed]:
                        break
                    kChoosed += 1
                if kChoosed == self.K:
                    self.K += 1
                    self.K_current += 1
            return kChoosed

        elif MODE == "iter" or MODE == "3":
            if self.hasIntialized[documentID] == 0:
                self.hasIntialized[documentID] = 1
            for k in range(self.K + 1):
                if prob_normalized[k] > self.threshold_fix:
                    self.hasIntialized[documentID] = 2
                    break
            kChoosed = 0
            if self.hasIntialized[documentID] == 2:
                bigPro = self.prob[0]
                for k in range(1, self.K + 1):
                    if self.prob[k] > bigPro:
                        bigPro = self.prob[k]
                        kChoosed = k
                if kChoosed == self.K:
                    self.K += 1
                    self.K_current += 1
            elif self.hasIntialized[documentID] == 1:
                for k in range(1, self.K + 1):
                    self.prob[k] += self.prob[k - 1]
                thred = random.random() * self.prob[self.K]
                while kChoosed < self.K + 1:
                    if thred < self.prob[kChoosed]:
                        break
                    kChoosed += 1
                if kChoosed == self.K:
                    self.K += 1
                    self.K_current += 1
            return kChoosed

        elif MODE == "last" or MODE == 4:
            if self.hasIntialized[documentID] == 0:
                self.hasIntialized[documentID] = 1
            for k in range(self.K + 1):
                if prob_normalized[k] > self.threshold_fix:
                    self.hasIntialized[documentID] = 2
                    break
            kChoosed = 0
            bigPro = self.prob[0]
            for k in range(1, self.K + 1):
                if self.prob[k] > bigPro:
                    bigPro = self.prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
            return kChoosed

        # 将只有一个样本的簇归入其它簇
        elif MODE == "del_one":
            if self.hasIntialized[documentID] == 0:
                self.hasIntialized[documentID] = 1
            for k in range(self.K + 1):
                if prob_normalized[k] > self.threshold_fix:
                    self.hasIntialized[documentID] = 2
                    break
            kChoosed = 0
            bigPro = self.prob[0]
            for k in range(1, self.K):
                if self.prob[k] > bigPro:
                    bigPro = self.prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
            return kChoosed

    def reComputeProbs(self, overflowCount_idx, K):
        for idx in range(len(self.Lambda)):
            max_overflow = -sys.maxsize
            for k in range(K+1):
                if overflowCount_idx[idx][k] > max_overflow and self.prob_idx[idx][k] > 0.0:
                    max_overflow = overflowCount_idx[idx][k]
            for k in range(K+1):
                if self.prob_idx[idx][k] > 0.0:
                    self.prob_idx[idx][k] = self.prob_idx[idx][k] * math.pow(self.largeDouble,
                                                                             overflowCount_idx[idx][k] - max_overflow)

    # Clear the useless cluster
    def checkEmpty(self, cluster):
        for idx in range(len(self.Lambda)):
            if cluster in self.n_z[idx] and self.m_z[cluster] == 0:
                if idx == len(self.Lambda) - 1:
                    self.K_current -= 1
                    self.m_z.pop(cluster)
                if cluster in self.n_z[idx]:
                    self.n_z[idx].pop(cluster)
                    self.n_zv[idx].pop(cluster)

    def output(self, documentSet, outputPath, wordList):
        outputDir = outputPath + self.dataset + self.ParametersStr + "/"
        try:
            isExists = os.path.exists(outputDir)
            print("yo:{}".format(isExists))
            if not isExists:
                print("i am here")
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        try:
            self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        except:
            print("\tOutput Phi Words Wrong!")
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):    # 后验估计φ
        self.phi_zv = [{} for _ in range(len(self.Lambda))]
        for idx in range(len(self.Lambda)):
            for cluster in self.n_zv[idx]:
                n_z_sum = 0
                if self.m_z[cluster] != 0:
                    if cluster not in self.phi_zv[idx]:
                        self.phi_zv[idx][cluster] = {}
                    for v in self.n_zv[idx][cluster]:
                        if self.n_zv[idx][cluster][v] != 0:
                            n_z_sum += self.n_zv[idx][cluster][v]
                    for v in self.n_zv[idx][cluster]:
                        if self.n_zv[idx][cluster][v] != 0:
                            self.phi_zv[idx][cluster][v] = float(self.n_zv[idx][cluster][v] +
                                                                 self.beta) / float(n_z_sum + self.beta0[idx])

    def getTop(self, array, rankList, Cnt):
        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in array:
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for idx in range(len(self.Lambda)):
            writer.write('\n--------------------- ' + str(idx) + ' -------------------\n\n')
            for k in range(self.K):
                rankList = []
                if k not in self.phi_zv[idx]:
                    continue
                topicline = "Topic " + str(k) + ":\n"
                writer.write(topicline)
                self.getTop(self.phi_zv[idx][k], rankList, Cnt)
                for i in range(rankList.__len__()):
                    tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[idx][k][rankList[i]])
                    writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if cluster in self.m_z and self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key=lambda tc: tc[1], reverse=True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):
        print("outputClusteringResult:{}".format(outputDir))
        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        for d in range(self.startDoc, self.currentDoc):
            documentID = documentSet.documents[d].documentID
            cluster = self.z[documentID] + self.All_K
            writer.write(str(documentID) + " " + str(cluster) + "\n")
        writer.close()


import json


class GSDPMM:

    def __init__(self, K, alpha, beta, iterNum, sampleNum, dataset, datasetNum, wordsInTopicNum, dataDir,
                 threshold_init, threshold_fix):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterNum = iterNum
        self.sampleNum = sampleNum
        self.dataset = dataset
        self.datasetNum = datasetNum
        self.wordsInTopicNum = wordsInTopicNum
        self.dataDir = dataDir
        self.threshold_init = threshold_init
        self.threshold_fix = threshold_fix

        self.All_K = 0
        self.authorList = []
        self.wordList = []
        self.wordToIdMap = {}
        self.result_dict = {}
        self.z = {}

    def getDocuments(self, author_no):
        self.documentSet = DocumentSet(self.dataDir + self.dataset + '-' + str(author_no), self.wordToIdMap, self.wordList)
        self.V = self.wordToIdMap.__len__()
    def runGSDPMM(self, sampleNo, outputPath):
        ParametersStr = "K" + str(self.K) + "iterNum" + str(self.iterNum) + \
                        "SampleNum" + str(self.sampleNum) + "alpha" + str(round(self.alpha, 4)) + \
                        "beta" + str(round(self.beta, 4)) + "InitThr" + str(self.threshold_init) + \
                        "FixThr" + str(self.threshold_fix)

        with open('data/WhoIsWho-authorList', 'r') as f:
            self.authorList = f.readline().split()
        for author_no in range(self.datasetNum):
            print('\n' + 'author disambiguation No.%d:' % author_no)
            # 运行模型时间太久了（1h），如想自行运行，请将下述注释取消
        
            author = self.authorList[author_no]
            if author == 'NoPaper':
                continue
            self.getDocuments(author_no)
            self.K = 0
            model = Model(self.K, self.All_K, self.V, self.iterNum, self.alpha, self.beta, self.threshold_init,
                          self.threshold_fix, self.dataset + '-' + str(author_no), ParametersStr, sampleNo,
                          self.wordsInTopicNum)
            self.z, self.All_K = model.run_GSDPMM(self.documentSet, outputPath, self.wordList)
            self.get_result_dict(self.documentSet, author)
    
        print('\n\n' + 'End of GSDPMM running, Start to generate json_result!')
        self.dump_json()
        print('Saving successful!')

    def get_result_dict(self, documentSet, author):
        paper_dict = {}
        for d in range(0, documentSet.D):
            documentID = documentSet.documents[d].documentID
            cluster = self.z[documentID]
            if str(cluster) not in paper_dict:
                paper_dict[str(cluster)] = [documentID]
            else:
                paper_dict[str(cluster)].append(documentID)
        self.result_dict[author] = list(paper_dict.values())
        
    def dump_json(self):
        json.dump(self.result_dict, open('result/WhoIsWho_res.json', 'w+', encoding='utf-8'), indent=4)
        
import time
import os
import argparse
import tensorflow as tf

K = 0  # 簇的初始个数
alpha = 0.1
beta = 0.05
iterNum = 5     # 吉布斯采样迭代次数
sampleNum = 1   # 算法采样次数
dataset = "WhoIsWho"
datasetNum = 50
wordsInTopicNum = 15    # 计算代表性词汇的top个数
dataDir = "data/"

threshold_init = 0      # 表示所有的文档都能被初始化
threshold_fix = 1.1     # 表示所有的prob都不fixed

outputPath = "result/"


def runGSDPMM(K, alpha, beta, iterNum, sampleNum, dataset, datasetNum, wordsInTopicNum, dataDir,
               threshold_init, threshold_fix):
    gsdpmm = GSDPMM(K, alpha, beta, iterNum, sampleNum, dataset, datasetNum, wordsInTopicNum,
                      dataDir, threshold_init, threshold_fix)
    for sampleNo in range(1, sampleNum + 1):
        print("Algorithm SampleNo:" + str(sampleNo))
        gsdpmm.runGSDPMM(sampleNo, outputPath)

def make_parser():
    parser = argparse.ArgumentParser('test network')
    parser.add_argument('-d', '--devices', default='0', type=str, help='device for testing')
    return parser

# 对聚好的簇中离散论文集基于余弦相似度进行合并（下面三步）

# 预处理结果簇
import codecs
import json
import warnings

# 不显示警告
warnings.filterwarnings("ignore")


class ClusterEvaluation:

    def __init__(self):
        self.labelsPred = {}
        self.labelsTrue = {}
        self.docs = []
        self.D = 0

    # Get labelsPred.
    def getMStreamPredLabels(self, inFile):
        with codecs.open(inFile, 'r') as fin:
            for line in fin:
                try:
                    documentID = line.strip().split()[0]
                    clusterNo = line.strip().split()[1]
                    self.labelsPred[documentID] = int(clusterNo)
                except:
                    print(line)

    # Get labelsTrue and docs.
    def getMStreamTrueLabels(self, dataset):
        with codecs.open(dataset, 'r') as fin:
            for docJson in fin:
                try:
                    docObj = json.loads(docJson)
                    self.labelsTrue[docObj['paperID']] = int(docObj['clusterNo'])
                    self.docs.append([docObj['paperID'], docObj['co-authors'], docObj['orgs'], docObj['title'],
                                      docObj['abstract'], docObj['keywords'], docObj['venue']])
                except:
                    print(docJson)

    def generateTrueAndPredLabels(self, outFile):
        Id2Newdata = {}
        Newdata_id = 0
        Newdata_text1 = []
        Newdata_text2 = []
        Newdata_text3 = []
        Newdata_textID = []
        Newdata_paperList = []
        Newdata_size = []
        Newdata_predictedCluster = []
        for i in range(len(self.docs)):
            documentID = self.docs[i][0]
            co_authors = self.docs[i][1]
            orgs = self.docs[i][2]
            keywords = self.docs[i][5]
            venue = self.docs[i][6]
            keywords_venue = keywords
            if venue != '':
                keywords_venue += ' ' + venue

            if documentID in self.labelsPred:
                predictedCluster = self.labelsPred[documentID]
                if predictedCluster not in Id2Newdata:
                    Id2Newdata[predictedCluster] = Newdata_id
                    Newdata_textID.append(documentID)
                    Newdata_predictedCluster.append(predictedCluster)
                    Newdata_text1.append(co_authors)
                    Newdata_text2.append(orgs)
                    Newdata_text3.append(keywords_venue)
                    Newdata_paperList.append(documentID)
                    Newdata_size.append(1)
                    Newdata_id += 1
                else:
                    Newdata_text1[Id2Newdata[predictedCluster]] += ' ' + co_authors
                    Newdata_text2[Id2Newdata[predictedCluster]] += ' ' + orgs
                    Newdata_text3[Id2Newdata[predictedCluster]] += ' ' + keywords_venue
                    Newdata_paperList[Id2Newdata[predictedCluster]] += ' ' + documentID
                    Newdata_size[Id2Newdata[predictedCluster]] += 1

        with codecs.open(outFile, 'w') as fout:
            for i in range(Newdata_id):
                docObj = {}
                docObj['textID'] = Newdata_textID[i]
                docObj['predictedCluster'] = Newdata_predictedCluster[i]
                docObj['size'] = Newdata_size[i]

                if Newdata_size[i] > 3:
                    # 1
                    count_dict = {}
                    for text in Newdata_text1[i].split():
                        if text not in count_dict:
                            count_dict[text] = 0
                        count_dict[text] += 1
                    up1 = []
                    for d in count_dict.items():
                        if d[1] > 1:
                            up1.append(d[0])
                    docObj['text'] = ' '.join(up1)
                    # 2
                    # count_dict = {}
                    # for text in Newdata_text2[i].split():
                    #     if text not in count_dict:
                    #         count_dict[text] = 0
                    #     count_dict[text] += 1
                    # up2 = []
                    # items = sorted(count_dict.items(), key=lambda k: k[1], reverse=True)
                    # m = 0
                    # for d in items:
                    #     # if m >= int(Newdata_size[i] / 3):
                    #     if m >= 3:
                    #         break
                    #     up2.append(d[0])
                    #     m += d[1]
                    # docObj['orgs'] = ' '.join(up2)
                    # 3
                    count_dict = {}
                    for text in Newdata_text3[i].split():
                        if text not in count_dict:
                            count_dict[text] = 0
                        count_dict[text] += 1
                    up3 = []
                    for d in count_dict.items():
                        if d[1] > 2:
                            up3.append(d[0])
                    docObj['keywords_venue'] = ' '.join(up3)
                else:
                    docObj['text'] = ' '.join(set(Newdata_text1[i].split()))
                    docObj['keywords_venue'] = ' '.join(set(Newdata_text3[i].split()))

                docObj['paperList'] = Newdata_paperList[i]
                docObj['orgs'] = ' '.join(set(Newdata_text2[i].split()))
                docJson = json.dumps(docObj)
                fout.write(docJson + '\n')

        self.D += Newdata_id

        self.labelsPred = {}
        self.labelsTrue = {}
        self.docs = []

    def generateNewauthorList(self, outFile, authorList):
        with open(outFile, 'w') as f:
            f.write(' '.join(authorList))


def MStreamMaxBatch():
    K = 0
    sampleNum = 1
    alpha = '0.03'
    beta = '0.05'
    iterNum = 5
    threshold_init = "0"
    threshold_fix = "1.1"
    dataset = "WhoIsWho"
    datasetNum = 50
    datasetPath = 'data/' + dataset
    inPath = 'result/'
    NewDataPath = 'NewData/'
    MStreamEvaluation = ClusterEvaluation()

    with open('data/WhoIsWho-authorList', 'r') as f:
        authorList = f.readline().split()
    for sampleNo in range(1, sampleNum + 1):
        MStreamEvaluation.labelsPred = {}
        # for name in iter
        for authorNo in range(datasetNum):
            if authorList[authorNo] == 'NoPaper':
                continue
            dirName = '%s-%dK%diterNum%dSampleNum%dalpha%sbeta%sInitThr%sFixThr%s/' % \
                      (dataset, authorNo, K, iterNum, sampleNum, alpha, beta, threshold_init, threshold_fix)
            inDir = inPath + dirName
            fileName = '%s-%dSampleNo%dClusteringResult.txt' % (dataset, authorNo, sampleNo)
            inFile = inDir + fileName
            MStreamEvaluation.getMStreamPredLabels(inFile)
            MStreamEvaluation.getMStreamTrueLabels(datasetPath + '-' + str(authorNo))

            MStreamEvaluation.generateTrueAndPredLabels(NewDataPath + "New_data-" + str(authorNo))

    MStreamEvaluation.generateNewauthorList(NewDataPath + 'New_data-authorList', authorList)

    print('All documents number:', str(MStreamEvaluation.D))
    
import json
import codecs
import os
from sklearn.metrics.pairwise import cosine_similarity


class Merging:

    def __init__(self, dataDir):
        self.D = 0

        self.author_count1 = 0
        self.author_dict1 = {}
        self.document1 = []
        self.author_count2 = 0
        self.author_dict2 = {}
        self.document2 = []
        self.author_count3 = 0
        self.author_dict3 = {}
        self.document3 = []

        self.wordFre = [{}, {}, {}]

        self.textID = []
        self.cluster = []
        self.cluster_list = []
        self.fixed = []
        self.K = 0
        self.paperList = []
        self.predictedCluster = []
        New_dataDir = dataDir
        with open(New_dataDir) as f:
            line = f.readline()
            while line:
                self.cluster.append(-1)
                self.fixed.append(0)
                obj = json.loads(line)
                self.textID.append(obj['textID'])
                # -------------------------------------------
                # not process size = 1
                # -------------------------------------------
                if obj['size'] == 1:
                    self.cluster[self.D] = self.K
                    self.cluster_list.append([self.D])
                    self.fixed[self.D] = 1
                    self.K += 1
                text1 = obj['text']
                for author in text1.split():
                    if author not in self.author_dict1:
                        self.wordFre[0][author] = 0
                        self.author_dict1[author] = self.author_count1
                        self.author_count1 += 1
                    self.wordFre[0][author] += 1
                self.document1.append(text1)

                text2 = obj['orgs']
                for org in text2.split():
                    if org not in self.author_dict2:
                        self.wordFre[1][org] = 0
                        self.author_dict2[org] = self.author_count2
                        self.author_count2 += 1
                    self.wordFre[1][org] += 1
                self.document2.append(text2)

                text3 = obj['keywords_venue']
                for keyword_venue in text3.split():
                    if keyword_venue not in self.author_dict3:
                        self.wordFre[2][keyword_venue] = 0
                        self.author_dict3[keyword_venue] = self.author_count3
                        self.author_count3 += 1
                    self.wordFre[2][keyword_venue] += 1
                self.document3.append(text3)

                self.paperList.append(obj['paperList'])
                self.predictedCluster.append(obj['predictedCluster'])
                self.D += 1
                line = f.readline()

    def get_cosine_similarity(self, text1, text2, wordFre_index):
        word1 = text1.split()
        word2 = text2.split()
        word_dict = {}
        word_id = 0
        word_list = []
        for w in word1:
            if w not in word_dict:
                word_dict[w] = word_id
                word_list.append(w)
                word_id += 1
        for w in word2:
            if w not in word_dict:
                word_dict[w] = word_id
                word_list.append(w)
                word_id += 1
        vec_cosine = [[0 for _ in range(word_id)] for _ in range(2)]
        for w in word1:
            vec_cosine[0][word_dict[w]] += 1
        for w in word2:
            vec_cosine[1][word_dict[w]] += 1
        # -------------------------------------------
        # wordCount / wordFre
        # -------------------------------------------
        for _ in range(word_id):
            vec_cosine[0][_] /= float(self.wordFre[wordFre_index][word_list[_]])
        return cosine_similarity(vec_cosine)[0][1]

    def process(self, sim_threshold):
        max_similarity = 0.0
        x = -1
        y = -1
        for i in range(self.D):
            if self.fixed[i] != 0:
                continue
            for j in range(i):
                if self.fixed[j] != 0:
                    continue
                w1 = 0.6
                w2 = 0.2
                w3 = 0.2
                similarity1 = self.get_cosine_similarity(self.document1[i], self.document1[j], 0)

                if self.document2[i].split() == [] or self.document2[j].split() == []:
                    w2 = 0.0
                    similarity2 = 0.0
                else:
                    similarity2 = self.get_cosine_similarity(self.document2[i], self.document2[j], 1)

                if self.document3[i].split() == [] or self.document3[j].split() == []:
                    w3 = 0.0
                    similarity3 = 0.0
                else:
                    similarity3 = self.get_cosine_similarity(self.document3[i], self.document3[j], 2)

                similarity = similarity1 * w1 + similarity2 * w2 + similarity3 * w3
                if similarity > max_similarity:
                    max_similarity = similarity
                    x = i
                    y = j
        # print(max_similarity)
        if max_similarity >= sim_threshold:
            if self.cluster[x] == -1 and self.cluster[y] == -1:
                self.cluster[x] = self.K
                self.cluster[y] = self.K
                self.fixed[y] = -1
                self.document1[x] += ' ' + self.document1[y]
                self.document2[x] += ' ' + self.document2[y]
                self.document3[x] += ' ' + self.document3[y]
                self.cluster_list.append([x, y])
                self.K += 1
            elif self.cluster[x] != -1 and self.cluster[y] != -1:
                x_choose = self.cluster[x]
                y_choose = self.cluster[y]
                self.cluster[y] = x_choose
                self.fixed[y] = -1
                self.document1[x] += ' ' + self.document1[y]
                self.document2[x] += ' ' + self.document2[y]
                self.document3[x] += ' ' + self.document3[y]
                self.cluster_list[x_choose] += self.cluster_list[y_choose]
                self.cluster_list[y_choose] = []
            else:
                if self.cluster[x] != -1:
                    x_choose = self.cluster[x]
                    self.cluster[y] = x_choose
                    self.fixed[y] = -1
                    self.document1[x] += ' ' + self.document1[y]
                    self.document2[x] += ' ' + self.document2[y]
                    self.document3[x] += ' ' + self.document3[y]
                    self.cluster_list[x_choose].append(y)
                else:
                    y_choose = self.cluster[y]
                    self.cluster[x] = y_choose
                    self.fixed[y] = -1
                    self.document1[y] += ' ' + self.document1[x]
                    self.document2[y] += ' ' + self.document2[x]
                    self.document3[y] += ' ' + self.document3[x]
                    self.cluster_list[y_choose].append(x)
                    # self.document1[y] = ' '.join(set(self.document1[y]))
                    # self.document2[y] = ' '.join(set(self.document2[y]))
                    # self.document3[y] = ' '.join(set(self.document3[y]))

            # self.document1[x] = ' '.join(set(self.document1[x]))
            # self.document2[x] = ' '.join(set(self.document2[x]))
            # self.document3[x] = ' '.join(set(self.document3[x]))
            return True
        return False

    def generate_result(self, sim_threshold, ouPath, dataset, sampleNo):
        while self.process(sim_threshold):
            continue
        cluster_merge_dict = {}
        cluster_merge_id = 0
        cluster_paperList = []
        cluster_predictedCluster = []
        for d in range(self.D):
            if self.fixed[d] == -1:
                continue
            if self.cluster[d] == -1:
                cluster = self.K
                self.cluster_list.append([d])
                self.K += 1
            else:
                cluster = self.cluster[d]
            for cluster_d in self.cluster_list[cluster]:
                if cluster not in cluster_merge_dict:
                    cluster_merge_dict[cluster] = cluster_merge_id
                    cluster_paperList.append(self.paperList[cluster_d])
                    cluster_predictedCluster.append(self.predictedCluster[cluster_d])
                    cluster_merge_id += 1
                else:
                    cluster_paperList[cluster_merge_dict[cluster]] += ' ' + self.paperList[cluster_d]

        outputDir = ouPath + dataset + "ClusteringResult/"
        try:
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        outputPath = outputDir + dataset + "SampleNo" + str(sampleNo) + "ClusteringResult.txt"
        with codecs.open(outputPath, 'w') as fout:
            for i in range(cluster_merge_id):
                docObj = {}
                docObj['predictedCluster'] = cluster_predictedCluster[i]
                docObj['paperList'] = cluster_paperList[i]
                docJson = json.dumps(docObj)
                fout.write(docJson + '\n')




if __name__ == '__main__':
    DataProcessing().generateDataset()
    outf = open("time_gsdpmm", "a")
    time1 = time.time()
    runGSDPMM(K, alpha, beta, iterNum, sampleNum, dataset, datasetNum, wordsInTopicNum, dataDir,
                threshold_init, threshold_fix)
    time2 = time.time()
    outf.write(str(dataset) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) +
               "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) +
               "InitThr" + str(threshold_init) + "FixThr" + str(threshold_fix) +
               "\ttime:" + str(time2 - time1) + "\n")
    datasetNum = 50
    dataDir = "NewData/"
    dataset = "New_data"
    ouPath = "result_merge/"
    sampleNum = 1
    sim_threshold = 0.048

    with open(dataDir + 'New_data-authorList', 'r') as f:
        authorList = f.readline().split()

    print('Starting Cluster mering !!!' + '\n')
    for author_no in range(datasetNum):
        # if author_no < 3:
        #     continue
        print('sim_threshold: ' + str(sim_threshold) + '\tCluster mering No.%d !' % author_no)
        author = authorList[author_no]
        if author == 'NoPaper':
            continue
        merge = Merging(dataDir + dataset + '-' + str(author_no))
        merge.generate_result(sim_threshold, ouPath, dataset + '-' + str(author_no), sampleNum)
    print('\n\n' + 'End of Merging running!')
    MStreamMaxBatch()


    
