import tensorflow as tf
from tensorflow.contrib import rnn

class BilinearAttentionO2P():

    def __init__(self,dim,vname):
        self.W = tf.get_variable(vname, [dim, dim],initializer=tf.contrib.layers.xavier_initializer())
        self.dim = dim

    def score(self,source,target):
        '''
        :param source: batch x num_option x emb_dim
        :param target: batch x emb_dim
        W -- emb_dim x emb_dim
        :return: batch x num_option
        '''
        target = tf.expand_dims(target,1)
        tmp = tf.tensordot(source,self.W,[[2],[0]])
        scores = tf.reduce_sum(tmp * target, axis=2)
        return tf.nn.softmax(scores)

class BilinearAttentionP2Q():

    def __init__(self,dim,vname):
        self.W = tf.get_variable(vname, [dim, dim], initializer=tf.contrib.layers.xavier_initializer())
        self.dim = dim

    def score(self,source,target):
        '''

        source -- batch x sentence_len x emb_dim
        target -- batch x emb_dim
        W -- emb_dim x emb_dim

        return batch x sentence_len
        '''
        source = tf.tensordot(source,self.W,[[2],[0]])
        tmp = tf.expand_dims(target,1)
        scores = tf.reduce_sum(source * tmp,axis=2)
        return tf.nn.softmax(scores)

class BilinearAttentionP2QA():

    def __init__(self,dim,vname):
        self.W = tf.get_variable(vname, [dim, dim], initializer=tf.contrib.layers.xavier_initializer())
        self.dim = dim

    def score(self,source,target):
        '''
        source -- batch x len x emb_dim
        target -- batch x question_num x emb_dim
        W -- emb_dim x emb_dim

        return batch x question_num x len
        '''

        source = tf.transpose(source,perm=[0,2,1])
        #return source
        tmp = tf.tensordot(target, self.W,[[2],[0]])
        scores =  tf.matmul(tmp, source)
        return tf.nn.softmax(scores)

class DotProductAttention():

    def __init__(self,dim,vname):
        self.W = tf.get_variable(vname, [dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.dim = dim

    def score(self,source):
        '''
        source -- batch x question_num x emb_dim
        W -- emb_dim x 1

        return batch x question_num
        '''
        tmp = tf.tensordot(source, self.W, [[2],[0]])
        tmp = tf.squeeze(tmp)
        return tf.nn.softmax(tmp)

