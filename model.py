import torch
from torch import nn
DEVICE = torch.cuda.device("cuda")

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size=128
        self.embedding_dim=128
        self.num_layers=3

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim, device='cuda')
        self.lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size, num_layers=self.num_layers, dropout=0.2)
        self.fc = nn.Linear(self.lstm_size, n_vocab)

        #self.device = 'cuda'
        self.device = torch.cuda.device("cuda")
        #self.to('cuda')
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        t1, t2 = (
        torch.zeros(self.num_layers, sequence_length, self.lstm_size), 
        torch.zeros(self.num_layers, sequence_length, self.lstm_size))
        t1.to('cuda')
        t2.to('cuda')
        return t1, t2