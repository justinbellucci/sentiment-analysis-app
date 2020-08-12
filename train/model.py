# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 07_31_2020                                  
# REVISED DATE: 

import torch.nn as nn
import torch

class LSTMClassifier(nn.Module):
    """ LSTM based RNN to perform sentiment analysis
    
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2):
        """ Initialize the model by setting up the various 
            layers.
        """
        super(LSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers)
        
#         self.dropout = nn.Dropout(0.2)
        
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):
        
        batch_size = x.size(0)
        x = x.t()
        reviews_lengths = x[0,:] # first row is the lenght of original review
        reviews = x[1:,:] # second row to end is each review
        
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        # stack lstm layers before they go into fully connected layer
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        fc_out = self.fc(lstm_out)

        sig_out = self.sigmoid(fc_out)
        # reshape sig_out (seq_length, batch_size, output_size)
        sig_out = sig_out.view(-1, batch_size, 1) 
        
        # here we want to grab the value of the output which
        # corresponds to the last word in the original sequence.
        # then we grab all in the batch. ex: out = sig_out[165, (0,50)]
        out = sig_out[reviews_lengths - 1, range(batch_size)]
        
        return out.squeeze() # squeeze dimensions

#     def init_hidden(self, batch_size):
#         """ Intialize the hidden state
#         """
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         weight = next(self.parameters()).data

#         hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
#                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

#         return hidden
   