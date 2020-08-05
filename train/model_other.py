import torch.nn as nn

class LSTMOther(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMOther, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
#         print(x.shape)
        lengths = x[0,:]
#         reviews = x[1:,:]
        embeds = self.embedding(x)
#         print(embeds.shape)
        lstm_out, _ = self.lstm(embeds)
#         print(lstm_out.shape)
        out = self.dense(lstm_out)
#         print(out.shape)
        out = out[lengths - 1, range(len(lengths))]
        sig_out = self.sig(out.squeeze())
#         print(sig_out.shape)
        
        return sig_out