#coding=utf8
"""
hoohaa
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

def length(sequences):
    # 返回序列中每一个元素的长度
    # # 输入inputs的shape是[batch_size, max_time, embedding_size] = [batch_size*sent_in_doc, word_in_sent, embedding_size]
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

class HAN():

    def __init__(self, vocab_size, num_classes, embedding_size=150, hidden_size=50):

        self.vocab_size = vocab_size#词表内的单词数
        self.num_classes = num_classes#种类数
        self.embedding_size = embedding_size#词嵌入的维度，也就是代表单词的向量的长度
        self.hidden_size = hidden_size#隐藏层的数目

        with tf.name_scope('placeholder'):#定义新的变量群‘placeholder’
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num') #初始化一个变量，执行时赋值，句子数目最大值
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')#句子长度最大值
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')#初始化记录每次投入的数据量
            # x的shape为[batch_size, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
            # y的shape为[batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')#初始化三维数组,每个格子代表一个单词
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')#初始化二维数组，有5列，应该是boolean型的

        # 构建模型
        word_embedded = self.word2vec()#把所有单词对应向量
        sent_vec = self.sent2vec(word_embedded)#根据单词与向量对应把所有句子对应向量
        doc_vec = self.doc2vec(sent_vec)#根据句子向量把所有文章对应向量
        out = self.classifer(doc_vec)#根据文本向量分类

        self.out = out#模型的输出结果，应该是12345中的一个


    def word2vec(self):
        with tf.name_scope("embedding"):#定义新的变量群embedding
            # 新建一个变量名叫embedding_mat，生成vocab行embedding列的矩阵，也就是词表那么大
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))
            # shape为[batch_size, sent_in_doc, word_in_sent, embedding_size]，
            #输入规模，每层代表一个输入的训练样本，然后每个样本用三维矩阵表示，文本中的句子数、句子中的单词数、单词长度
            #word_embedded应该是对应这个数据的词表,应该是把文本里的单词用词表向量表示
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
        return word_embedded

    def sent2vec(self, word_embedded):
        # 输入shape为[batch_size, sent_in_doc, word_in_sent, embedding_size]
        #word_embedded应该相当于把文章里的每个单词用词向量表示了
        with tf.name_scope("sent2vec"):#定义命名域
            # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            # batch_size * sent_in_doc当做是batch_size.这样一来，每次输入一个句子，每个GRU的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            # shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
            #将四维降为三维，即以文章为元素变为以句子为元素
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])
            # shape为[batch_size*sent_in_doc, word_in_sent, hidden_size*2]
            #GRU里面有hiddensize*2个格子，是将每个单词转化为长度为hidden_size*2的数据
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')#编码
            # shape为[batch_size*sent_in_doc, hidden_size*2]
            # 对每个句子进行attention处理
            sent_vec = self.AttentionLayer1(word_encoded, name='word_attention')#赋权
            return sent_vec

    def doc2vec(self, sent_vec):
        # 输入shape为[batch_size*sent_in_doc, hidden_size*2]
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])
            # shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            # shape为[batch_szie, hidden_szie*2]
            doc_vec = self.AttentionLayer2(doc_encoded, name='sent_attention')
            return doc_vec

    def classifer(self, doc_vec):
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            return out

    def BidirectionalGRUEncoder(self, inputs, name):
        # 输入inputs的shape是[batch_size, max_time, embedding_size] = [batch_size*sent_in_doc, word_in_sent, embedding_size]
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            # outputs的size是[batch_size, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs
            
    def AttentionLayer1(self, inputs, name):
                # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
                    # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
                    # 因为使用双向GRU，所以其长度为2×hidden_szie
                    # 新建一个hidden_size*2这么大的数组，是attention向量
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
                    # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.relu)
                    # 1处理下来把每个单词变成一个值
                    # shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
                   # hard_alpha = tf.Print(alpha, [alpha])
            self.one_alpha = alpha

                    # reduce_sum之前shape为[batch_size, max_time, hidden_size*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output


    def AttentionLayer2(self, inputs, name):
        # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie
            # 新建一个hidden_size*2这么大的数组，是attention向量
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.relu)
            # 1处理下来把每个单词变成一个值
            # shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
           # soft_alpha = tf.Print(alpha, [alpha])
            self.two_alpha = alpha
            # reduce_sum之前shape为[batch_size, max_time, hidden_size*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output

