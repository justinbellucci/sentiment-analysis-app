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
        
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=False)
        
        self.dropout = nn.Dropout(0.2)
        
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        """ Perform a forward pass of our model on some input and 
            hidden state.
        """
        batch_size = x.size(0)
        x = x.t()[1:,:]                      # INPUT:  (seq_length, batch_size)
#         print("Input shape: {}".format(x.shape))
        embeds = self.embedding(x)        # (seq_length, batch_size, n_embed)
#         print("Embed shape: {}".format(embeds.shape))
        lstm_out, hidden = self.lstm(embeds, hidden)   # (seq_length, batch_size, n_hidden)
#         print("lstm shape: {}".format(lstm_out.shape))
        
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) # (batch_size*seq_length, n_hidden)
#         print("lstm stacked shape: {}".format(lstm_out.shape))
        out = self.dropout(lstm_out)
        out = self.fc(out)                # (batch_size*seq_length, n_output)
        sig_out = self.sigmoid(out)       # (batch_size*seq_length, n_output)
#         print("sig_out shape: {}".format(sig_out.shape))
        sig_out = sig_out.view(batch_size, -1)    # (batch_size, seq_length*n_output)
        sig_out = sig_out[:, -1]                  # (batch_size)
#         print(sig_out.shape)
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """ Intialize the hidden state
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden
   