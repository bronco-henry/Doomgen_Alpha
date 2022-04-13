import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import Model
from dataset import Dataset
from train import train, predict

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sequence_length', type=int, default=4)
args = parser.parse_args()

model = Model(Dataset(128))
dataset = Dataset(args)

train(dataset, model, args)

        #note: 'text' will be used as a key so must only contain 
        #       characters that are in the input set
print(predict(dataset, model, text = 'doom says'))
