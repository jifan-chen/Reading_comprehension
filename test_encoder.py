import torch
from torch import nn
from utils import *
from torch.autograd import Variable


class RNNEncoder(nn.Module):
    """ The standard RNN encoder. """
    def __init__(self, input_size, rnn_type, bidirectional, num_layers, batch_first,
                 hidden_size , dropout=0.5):
        super(RNNEncoder, self).__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.rnn_forward = getattr(nn, rnn_type)(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first = batch_first,
                #dropout = dropout
                )
        for param in self.rnn_forward.parameters():
            nn.init.normal(param,0,0.1)
        self.rnn_backward = getattr(nn, rnn_type)(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first = batch_first,
                #dropout = dropout
                )
        for param in self.rnn_backward.parameters():
            nn.init.normal(param,0,0.1)

    def inverse_input(self,input,last_position):
        new_input = input.clone()
        for i in range(input.data.shape[0]):
            idx = int(last_position[i].data.numpy())
            inv_idx = torch.arange(idx,-1,-1).long()
            #print inv_idx
            tmp = input[i].clone()
            inverse = tmp[inv_idx,:]
            new_input[i,:idx+1,:] = inverse
            #print input
        return new_input

    def forward(self, input, mask, last_position):

        if self.bidirectional:
            outputs_forward, hidden_t_forward = self.rnn_forward(input)
            masked_outputs_forward = outputs_forward * mask
            hidden_t_forward = outputs_forward[range(outputs_forward.shape[0]), last_position, :]

            #print input
            inverse_input = self.inverse_input(input,last_position)

            outputs_backward, hidden_t_backward = self.rnn_backward(inverse_input)
            masked_outputs_backward = outputs_backward * mask
            hidden_t_backward = outputs_backward[range(outputs_backward.shape[0]), last_position, :]

            hidden_t = torch.cat((hidden_t_forward,hidden_t_backward),1)
            outputs = torch.cat((masked_outputs_forward,masked_outputs_backward),2)

        else:
            outputs,hidden_t = self.rnn_forward(input)
            outputs = outputs * mask
            try:
                hidden_t = outputs[range(outputs.shape[0]),last_position,:]
            except IndexError:
                print last_position
                print outputs.data
        return hidden_t, outputs

if __name__ == '__main__':
    t = [[1, 2, 3], [1], [1, 2, 3, 4], [3, 4, 5, 6, 7]]
    emb_size = 10
    padded_sentences, masks = pad_sequences(t, padding='post', truncating='post')
    embeddings = Variable(torch.randn(4,5,emb_size))
    print embeddings.data.shape
    rnn = RNNEncoder(input_size=10,rnn_type='LSTM',num_layers=1,bidirectional=False,
                     batch_first=True,hidden_size=15)
    masks.mask = torch.from_numpy(masks.mask).type(torch.FloatTensor)

    ht,outputs = rnn(embeddings,masks)
    print ht[0]
    print outputs.data[0]