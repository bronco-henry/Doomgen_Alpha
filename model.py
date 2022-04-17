import torch
from torch import nn
import os
import pickle
from pathlib import Path

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        
        #self.MODELFOLDER = "./models"
        #self.MODELPATH = os.path.join(self.MODELFOLDER,"model.pt")
        self.MODELPATH = Path("./models/model.pt")
        self.EPOCHSPATH = Path("./models/modelepochs.pickle")
        self.DEFAULTSTATE = {"Epochs": 0, "Name": "DOOMBOT"}

        self.model_data = self.DEFAULTSTATE.copy()
        self.epochs_trained = self.model_data["Epochs"]
        self.model_name = self.model_data["Name"]

        if os.path.exists(self.EPOCHSPATH):
            #print("DEBUGDEBUGDEBUGDEBUGDEBUG: Attempting to load number of prev epochs")
            self.load_epochs()
        else:
            print(">>File containing prev. epochs does not exist. File will be created after initial training or with SAVE.")

        self.lstm_size=128
        self.embedding_dim=128
        self.num_layers=3

        self.cuda0 = torch.device('cuda:0')

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim, device=self.cuda0)
        self.lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size, num_layers=self.num_layers, dropout=0.2, device=self.cuda0)
        self.fc = nn.Linear(self.lstm_size, n_vocab, device=self.cuda0)



    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state


    def save_epochs(self):
        #torch.save(self.state_dict(), self.MODELPATH)
        print("DEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUG: TRYING TO WRITE CURRENT OF EPOCHS: ", (self.epochs_trained))
        with open(self.EPOCHSPATH, 'wb+') as self.f:
            self.model_data["Epochs"] = self.epochs_trained
            self.f.seek(0)
            self.f.truncate()
            pickle.dump(self.model_data, self.f, pickle.HIGHEST_PROTOCOL)
            self.f.close()
        return


    def load_epochs(self):
        with open(self.EPOCHSPATH, 'rb') as self.f:
            self.model_data = pickle.load(self.f)
            print("DEBUGDEBUGDEBUGDEBUGDEBUGDEBUG: ", self.model_data)
            self.epochs_trained = self.model_data["Epochs"]
        return 


    def init_state(self, sequence_length):
        return (
        torch.zeros(self.num_layers, sequence_length, self.lstm_size, device=self.cuda0), 
        torch.zeros(self.num_layers, sequence_length, self.lstm_size, device=self.cuda0)
        )

if __name__ == "__main__":
    print("why was this run as main lol")