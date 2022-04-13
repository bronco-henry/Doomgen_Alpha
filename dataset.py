import torch 
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indices = [self.word_to_index[w] for w in self.words]
        
    def load_words(self):
        print("\nLoading words....")
        train_df = open('doomwordlist.txt', 'r', encoding='utf-8').read()
        return train_df.split(" ")

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indices) - self.args.sequence_length

    def __getitem__(self, index):
        cuda0 = torch.device('cuda:0')

        t1 = torch.tensor(self.words_indices[index:index+self.args.sequence_length], device=cuda0)
        #print(t1.get_device())
        t2 = torch.tensor(self.words_indices[index+1:index+self.args.sequence_length+1], device=cuda0)
        #t2.device = 'cuda'

        #print(t1.device, t2.device)
        return(
            t1, 
            t2
        )