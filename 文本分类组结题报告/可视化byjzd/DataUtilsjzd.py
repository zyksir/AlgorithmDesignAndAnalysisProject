#coding=utf-8
"""
hoohaa
"""
import json
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
import numpy as np

# 使用nltk分词分句器
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  # 加载英文的划分句子的模型
word_tokenizer = WordPunctTokenizer()  # 加载英文的划分单词的模型

# 记录每个单词及其出现的频率
word_freq = defaultdict(int)  # Python默认字典

# 读取数据集，并进行分词，统计每个单词出现次数，保存在word freq中
with open('mydata.json', 'rb') as f:
    for line in f:
        review = json.loads(line)
        words = word_tokenizer.tokenize(review['text'])  # 将评论句子按单词切割
        for word in words:
            word_freq[word] += 1
    print("load finished")

# 将词频表保存下来
with open('word_freq.pickle', 'wb') as g:
    pickle.dump(word_freq, g)
    print(len(word_freq))  # 159654
    print("word_freq save finished")

num_classes = 5
sort_words = list(sorted(word_freq.items(), key=lambda x: -x[1]))  # 按出现频数降序排列
print (sort_words[:10], sort_words[-10:] ) # 打印前十个和倒数后十个

# 构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOW
vocab = {}
i = 1
vocab['UNKNOW_TOKEN'] = 0
for word, freq in word_freq.items():
    if freq > 5:
        vocab[word] = i
        i += 1
print (i)  # 46960

UNKNOWN = 0
data_x = []
data_y = []
data_z = []#jzd
max_sent_in_doc = 50
max_word_in_sent = 50

# 将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
# 多余的删除，并保存到最终的数据集文件之中
# 注意，少于30的并没有进行补零操作
with open('mydata.json', 'rb') as f:
    for line in f:
        # doc = []
        doc = np.zeros((50, 50), dtype=np.int32)
        my_doc = []
        review = json.loads(line)
        sents = sent_tokenizer.tokenize(review['text'])  # 将评论分句
        for i, sent in enumerate(sents):
            if i < max_sent_in_doc:
                word_to_index = []
                my_doc_mid = []#jzd
                for j, word in enumerate(word_tokenizer.tokenize(sent)):
                    if j < max_word_in_sent:
                        # word_to_index.append(np.array(vocab.get(word, UNKNOWN)))
                        doc[i][j] = vocab.get(word, UNKNOWN)
                        my_doc_mid.append(word)#jzd
                # doc.append(np.array(word_to_index))
                my_doc.append(my_doc_mid)

        label = int(review['type'])
        labels = [0] * num_classes
        labels[label-1] = 1
        data_y.append(labels)
        data_x.append(doc.tolist())
        data_z.append(my_doc)#jzd
    pickle.dump((data_x, data_y), open('myproject_data', 'wb'))
    pickle.dump(data_z, open('kshfromjzd', 'wb'))#jzd

    print("data_X:", data_x[10])
    print("data_y:", data_y[10])
    print("data_z:", data_z[3976])#jzd
    print (len(data_x) ) # 229907
    # length = len(data_x)
    # train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1 :]
    # train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1 :]
    
