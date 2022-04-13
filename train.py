import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

def train(dataset, model, args):
    mydataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state_h, state_c, = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(mydataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1,2), y)
            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss':loss.item() })

# TODO: make this a more 'interactive' function perhaps
def predict(dataset, model, text, next_words = 100):
    model.eval()
    print(model.eval())

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        cuda0 = torch.device('cuda:0')
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]], device=cuda0)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return " ".join(words)
