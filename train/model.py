# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 07_31_2020                                  
# REVISED DATE: 

import torch.nn as nn

class LSTMClassifier(nn.Module):
    """ LSTM based RNN to perform sentiment analysis
    
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers=1, drop_prob=0.5):
        """ Initialize the model by setting up the various 
            layers.
        """
        super(LSTMClassifier, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.word_dict = None

    def forward(self, x, hidden):
        """ Perform a forward pass of our model on some input and 
            hidden state.
        """
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)    
        sig_out = self.sigmoid(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size, device):
        """ Intialize the hidden state
        """
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden