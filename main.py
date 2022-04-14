import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import Model
from dataset import Dataset
from train import train, predict, eval

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=11)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sequence_length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)
model = Model(dataset)

train(dataset, model, args)
print(eval(model))
print("\nTraining complete. Saving model.")

# TODO: save model here
 
text = ""
while text!= "QUIT":
    text = input("\nEnter some shit to base the model off of. Might fail btw. Type QUIT to end\n>")
 
        #note: 'text' will be used as a key so must only contain 
        #       characters that are in the input set
    try:
        print(predict(dataset, model, text, next_words = 50))
    except KeyError:
        if text=="QUIT":
            break
        else:
            print("\nBad key, try another input sequence please.")

print("\nexiting!")