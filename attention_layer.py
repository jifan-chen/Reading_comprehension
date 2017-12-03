import torch
import torch.nn as nn
from torch.autograd import Variable

class BilinearAttentionO2P(nn.Module):

    def __init__(self,dim):
        super(BilinearAttentionO2P, self).__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.randn(dim, dim))
        nn.init.uniform(self.W,-0.01,0.01)
        self.sm = nn.Softmax(dim=1)

    def score(self,source,target):
        '''
        :param source: batch x num_option x emb_dim
        :param target: batch x emb_dim
        :return: batch x num_option
        '''

        target = torch.unsqueeze(target,2)
        tmp = torch.matmul(source,self.W)
        return torch.bmm(tmp,target)

    def forward(self, source, target):
        align = self.score(source, target)
        align = torch.squeeze(align,2)
        align_normalize = self.sm(align)
        return align_normalize

class BilinearAttentionP2Q(nn.Module):

    def __init__(self,dim):
        super(BilinearAttentionP2Q, self).__init__()
        self.dim =dim
        self.W = nn.Parameter(torch.randn(dim,dim))
        nn.init.uniform(self.W, -0.01, 0.01)
        self.sm = nn.Softmax(dim=1)

    def score(self,source,target):
        '''

        source -- batch x sentence_len x emb_dim
        target -- batch x emb_dim
        W -- emb_dim x emb_dim

        return batch x sentence_len x 1
        '''
        source = torch.matmul(source,self.W)
        tmp = torch.unsqueeze(target,2)
        return torch.bmm(source, tmp)

    def forward(self, source, target):
        align = self.score(source, target)
        align_normalize = self.sm(torch.squeeze(align))
        return align_normalize

class BilinearAttentionP2QA(nn.Module):

    def __init__(self,dim):
        super(BilinearAttentionP2QA, self).__init__()
        self.dim =dim
        self.W = nn.Parameter(torch.randn(dim,dim))
        nn.init.uniform(self.W, -0.5, 0.5)
        self.sm = nn.Softmax(dim=2)

    def score(self,source,target):
        '''
        source -- batch x len x emb_dim
        target -- batch x question_num x emb_dim
        W -- emb_dim x emb_dim

        return batch x question_num x len
        '''

        source = torch.transpose(source,1,2)
        tmp = torch.matmul(target, self.W)
        return torch.bmm(tmp, source)

    def forward(self, source, target):
        align = self.score(source, target)
        align_normalize = self.sm(align)
        return align_normalize

class DotProductAttention(nn.Module):

    def __init__(self,dim):
        super(DotProductAttention, self).__init__()
        self.dim =dim
        self.W = nn.Parameter(torch.randn(dim,1))
        nn.init.uniform(self.W, -0.5, 0.5)
        self.sm = nn.Softmax(dim=1)

    def score(self,source):
        '''
        source -- batch x question_num x emb_dim
        W -- emb_dim x 1

        return batch x question_num
        '''
        tmp = torch.matmul(source, self.W)
        return tmp

    def forward(self, source):
        align = self.score(source)
        align = torch.squeeze(align)
        align_normalize = self.sm(align)
        return align_normalize

if __name__ == '__main__':
    a = Variable(torch.randn(5,10,20))
    b = Variable(torch.randn(5,20))
