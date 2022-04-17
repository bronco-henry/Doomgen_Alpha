import torch 
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.train_words = self.load_words()
        self.validation_words = []
        self.test_words = []

        self.uniq_words = self.get_uniq_words()
        self.cuda0 = torch.device('cuda:0')

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.train_words_indices = [self.word_to_index[w] for w in self.train_words]
        
    def load_words(self):
        print("\nLoading words....")
        train_df = open('doomwordlist.txt', 'r', encoding='utf-8').read()
        return train_df.split(" ")

    def get_uniq_words(self):
        word_counts = Counter(self.train_words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.train_words_indices) - self.args.sequence_length

    def __getitem__(self, index):

       return (
        torch.tensor(self.train_words_indices[index:index+self.args.sequence_length], device=self.cuda0),
        torch.tensor(self.train_words_indices[index+1:index+self.args.sequence_length+1], device=self.cuda0)
        )
        
if __name__ == "__main__":
    print("why was this run as main lol")