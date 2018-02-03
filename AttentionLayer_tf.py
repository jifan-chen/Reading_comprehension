import tensorflow as tf
from tensorflow.contrib.layers import layer_norm
from Encoder_tf import *
from tensorflow.contrib import rnn

class BilinearAttentionO2P():

    def __init__(self,dim,vname):
        self.W = tf.get_variable(vname, [dim, dim],initializer=tf.contrib.layers.xavier_initializer())
        #self.W = tf.get_variable(vname, [dim, dim], initializer=tf.random_normal_initializer(-0.01,0.01))
        self.dim = dim

    def score(self,source,target):
        '''
        :param source: batch x num_option x emb_dim
        :param target: batch x emb_dim
        W -- emb_dim x emb_dim
        :return: batch x num_option
        '''
        tmp = tf.matmul(target,self.W)
        tmp = tf.expand_dims(tmp,1)
        scores = tf.reduce_sum(source * tmp, axis=2)
        return scores

class BilinearAttentionP2Q():

    def __init__(self,dim,vname):
        self.W = tf.get_variable(vname, [dim, dim], initializer=tf.contrib.layers.xavier_initializer())
        #self.W = tf.get_variable(vname, [dim, dim], initializer=tf.random_normal_initializer(-0.01, 0.01))
        self.dim = dim

    def score(self,source,target):
        '''
        source -- batch x sentence_len x emb_dim
        target -- batch x emb_dim
        W -- emb_dim x emb_dim

        return batch x sentence_len
        '''
        tmp = tf.matmul(target, self.W)
        tmp = tf.expand_dims(tmp, 1)
        scores = tf.reduce_sum(source * tmp, axis=2)
        return tf.nn.softmax(scores)

class BilinearAttentionP2QA():

    def __init__(self,dim,vname):
        self.W = tf.get_variable(vname, [dim, dim], initializer=tf.contrib.layers.xavier_initializer())
        #self.W = tf.get_variable(vname, [dim, dim], initializer=tf.random_normal_initializer(-0.01, 0.01))
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
        self.W = tf.get_variable(vname, [dim, dim], initializer=tf.contrib.layers.xavier_initializer())
        #self.W = tf.get_variable(vname, [dim, dim], initializer=tf.random_normal_initializer(-0.01, 0.01))
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

class BilinearDotM2M():

    def __init__(self,dim,vname):
        self.W = tf.get_variable(vname, [dim, dim], initializer=tf.contrib.layers.xavier_initializer())
        self.dim = dim

    def score(self,s1,s2):
        '''
        s1 -- batch x length x dim
        s2 -- batch x length x dim
        W -- dim x dim
        return batch x length
        '''
        tmp = tf.tensordot(s1,self.W,[[2],[0]])
        score = tf.reduce_sum(tmp * s2,axis=2)
        return score

class BilinearAttentionM2M():

    def __init__(self,dim,vname):
        self.W = tf.get_variable(vname, [dim, dim], initializer=tf.contrib.layers.xavier_initializer())
        self.dim = dim

    def score(self,s1,s2,mask):
        '''
        s1 -- batch x length x dim
        s2 -- batch x length x dim
        W -- dim x dim
        return batch x length
        '''
        tmp = tf.tensordot(s1,self.W,[[2],[0]])
        score = tf.reduce_sum(tmp * s2,axis=2)
        return score
class SelfAttention():

    def __init__(self,dim,vname):
        self.Wv = tf.get_variable(vname+'Wv',[dim,dim],initializer=tf.contrib.layers.xavier_initializer())
        self.Wq = tf.get_variable(vname+'Wq',[dim,dim],initializer=tf.contrib.layers.xavier_initializer())
        self.Wk = tf.get_variable(vname+'Wk',[dim,dim],initializer=tf.contrib.layers.xavier_initializer())
        self.W1fn = tf.get_variable(vname+'W1',[dim,dim*2],initializer=tf.contrib.layers.xavier_initializer())
        self.b1fn = tf.get_variable(vname+'b1',[dim*2],initializer=tf.random_uniform_initializer(-0.1,0.1))
        self.W2fn = tf.get_variable(vname+'W2',[dim*2,dim],initializer=tf.contrib.layers.xavier_initializer())
        self.b2fn = tf.get_variable(vname+'b2',[dim],initializer=tf.random_uniform_initializer(-0.1,0.1))
        self.dim = dim

    def apply_self_attention(self,s):
        query = tf.tensordot(s,self.Wq,[[2],[0]])
        key = tf.tensordot(s, self.Wk, [[2],[0]])
        value = tf.tensordot(s, self.Wv, [[2],[0]])
        att = tf.reduce_sum(query * key,axis=2) / tf.sqrt(tf.cast(self.dim,tf.float32))
        norm_att = tf.nn.softmax(att)
        norm_att = tf.expand_dims(norm_att,2)
        attended_output = value * norm_att + s

        # ffn
        #ffn1 = tf.nn.relu(tf.tensordot(attended_output,self.W1fn,[[2],[0]]))
        #ffn2 = tf.tensordot(ffn1,self.W2fn,[[2],[0]])
        return attended_output

class GatedAttention():
    def __init__(self,hidden_dim,weight_dim,vname):
        self.W = tf.get_variable(vname+'W',[weight_dim,weight_dim],initializer=tf.contrib.layers.xavier_initializer())
        #self.encoder = RNN_encoder(hidden_dim,vname+'encoder',bidirectional=True)

    def apply_attention(self,query,value,q_mask):
        '''
        :param query: batch x q_length x dim
        :param value: batch x d_length x dim
        :param W: dim x dim
        :return: batch x length x dim
        '''
        qt = tf.tensordot(query,self.W,[[2],[0]])
        qt = tf.transpose(qt,[0,2,1])
        alpha = tf.nn.softmax(tf.matmul(value,qt))
        alpha = alpha * tf.expand_dims(q_mask,1)
        alpha = alpha / tf.reduce_sum(alpha,axis=2,keep_dims=True)
        qhat = tf.matmul(alpha,query)

        dgated = qhat * value
        return dgated

class GatedAttentionWithOption():
    def __init__(self,hidden_dim,weight_dim,vname):
        self.W = tf.get_variable(vname+'W',[weight_dim,weight_dim],initializer=tf.contrib.layers.xavier_initializer())
        #self.encoder = RNN_encoder(hidden_dim,vname+'encoder',bidirectional=True)

    def apply_attention(self,query,value,q_mask):
        '''
        :param query: batch x num x o_length x dim
        :param value: batch x num x d_length x dim
        :param W: dim x dim
        :return: batch x length x dim
        '''
        qt = tf.tensordot(query,self.W,[[3],[0]])
        qt = tf.transpose(qt,[0,1,3,2])
        alpha = tf.nn.softmax(tf.matmul(value,qt))
        alpha = alpha * tf.expand_dims(q_mask,2)
        alpha = alpha / tf.reduce_sum(alpha,axis=3,keep_dims=True)
        qhat = tf.matmul(alpha,query)

        dgated = qhat * value
        return dgated

class MultihopAttention():
    def __init__(self,dim,vname):
        self.Wv = tf.get_variable(vname+'Wv',[dim,dim],initializer=tf.contrib.layers.xavier_initializer())
        self.Wq = tf.get_variable(vname+'Wq',[dim,dim],initializer=tf.contrib.layers.xavier_initializer())
        self.Wk = tf.get_variable(vname+'Wk',[dim,dim],initializer=tf.contrib.layers.xavier_initializer())
        self.W1fn = tf.get_variable(vname+'W1',[dim,dim*2],initializer=tf.contrib.layers.xavier_initializer())
        self.b1fn = tf.get_variable(vname+'b1',[dim*2],initializer=tf.random_uniform_initializer(-0.1,0.1))
        self.W2fn = tf.get_variable(vname+'W2',[dim*2,dim],initializer=tf.contrib.layers.xavier_initializer())
        self.b2fn = tf.get_variable(vname+'b2',[dim],initializer=tf.random_uniform_initializer(-0.1,0.1))
        self.dim = dim

    def apply_self_attention(self,s1,s2):
        query = tf.tensordot(s2,self.Wq,[[2],[0]])
        key = tf.tensordot(s1, self.Wk, [[2],[0]])
        value = tf.tensordot(s1, self.Wv, [[2],[0]])
        att = tf.reduce_sum(query * key,axis=2) / tf.sqrt(tf.cast(self.dim,tf.float32))
        norm_att = tf.nn.softmax(att)
        norm_att = tf.expand_dims(norm_att,2)
        attended_output = layer_norm(value * norm_att + s1)

        # ffn
        ffn1 = tf.nn.relu(tf.tensordot(attended_output,self.W1fn,[[2],[0]]))
        ffn2 = tf.tensordot(ffn1,self.W2fn,[[2],[0]])
        output = layer_norm(ffn2 + attended_output)
        return output