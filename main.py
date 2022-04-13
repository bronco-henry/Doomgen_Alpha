#woe is me. Trying again.

import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import Model
from dataset import Dataset
from train import train, predict
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

model = Model(Dataset(128))
dataset = Dataset(args)

train(dataset, model, args)
print(predict(dataset, model, text = 'doom is here to'))