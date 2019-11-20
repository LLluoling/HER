# coding:utf8

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from policy import extract_sent_compute_loss


torch.manual_seed(233)


class SHE(nn.Module):
    def __init__(self, config):
        super(SHE, self).__init__()   
        
        # Parameters
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim
        self.word_input_size = config.word_input_size  # 100
        self.sent_input_size = config.sent_input_size  # 400
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units  # 200
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units  # 200
        self.num_layers = config.num_layers # 2 layers
        self.num_directions =config.num_directions # bidirectional
        self.pooling_way=config.pooling_way # way to generate overall info
        self.fixed_length =config.fixed_length
        self.batch_size = config.batch_size
        self.novelty = config.novelty

        # Parameters of Classification Layer
        # all 4h * 4h
        self.parameter_dim = 2*self.sent_LSTM_hidden_units
        
        # Network
        self.drop = nn.Dropout(self.dropout)
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        
        
        # local net
        self.kernel_sizes = [int(i) for i in config.filter_sizes.split(',')]
        self.num_filters = config.num_filters  #50 = h/2 or 100 = h
        self.convs = nn.ModuleList([nn.Conv2d(1,self.num_filters,(i,2*self.sent_LSTM_hidden_units))  for i in self.kernel_sizes])
        self.Wl = [Parameter(torch.randn(self.parameter_dim, self.num_filters)) for _ in range(len(self.kernel_sizes))]

        # word_LSTM output 400; sent_LSTM output 400
        self.word_LSTM = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True) 
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        
        self.in_dim = 2*self.parameter_dim + 3*self.num_filters
        self.decoder = nn.Sequential(nn.Linear(self.in_dim, 200),
                                         nn.Tanh(),
                                         nn.Linear(200, 1),
                                         nn.Sigmoid())

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)
    
    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, h)
        # word_features = self.drop(word_features)
        # word_outputs (N,W,4h)
        word_outputs, _ = self.word_LSTM(word_features) 
        # sent_features:(1,N,4h)
        sent_features = self._avg_pooling(word_outputs, sequence_length).view(1, sequence_num,self.sent_input_size)  
        sent_features = self.drop(sent_features)

        # sentence level LSTM
        # sent_outputs:(1,N,4h);  h_n, c_n (4,1,2h)  (num_layers * num_directions, batch, hidden_size)
        sent_outputs, _ = self.sent_LSTM(sent_features) 
        sent_outputs = self.drop(sent_outputs)
        
        ########  general representation  ####### 
        doc_features = sent_outputs.mean(dim=1)
        
        ########  local representations  ####### 
        sent_enc_tmp = sent_outputs.unsqueeze(1) #(1,1,N,4h)
        # conv_res 3 * (1, h/2, N-filter_size+1)   (1, num_filters=2h, N-filter_size+1)
        conv_res = [F.relu(conv(sent_enc_tmp)).squeeze(3) for conv in self.convs]
        # pool_res 3* (1, h/2) 
        pool_res = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in conv_res]
        self.local_outputs = pool_res
        
        
        self.sent_outputs = sent_outputs.squeeze(0) # (N,4h) 
        self.doc_outputs = doc_features.squeeze(0).expand(self.sent_outputs.size())
        local_outputs_tmp = [local_output.squeeze(0).expand((sequence_num,local_output.size()[-1])) 
                            for local_output in self.local_outputs] 

        enc_output = torch.cat([self.sent_outputs, self.doc_outputs,local_outputs_tmp[0],
                        local_outputs_tmp[1],local_outputs_tmp[2]], dim=-1)
        
        prob = self.decoder(enc_output).view(sequence_num, 1)

        return prob.view(sequence_num, 1)

