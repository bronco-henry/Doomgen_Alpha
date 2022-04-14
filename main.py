import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import Model
from dataset import Dataset
from train import train, predict, eval
import os

MODELFOLDER = "./models"
MODELPATH = os.path.join(MODELFOLDER,"model.pt")

# TODO: get rid of argparse - this is unneccessary and batch_size is not being used correctly I THINK
parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sequence_length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)
model = Model(dataset)
try: 
    model.load_state_dict(torch.load(MODELPATH))
    print("\n Model found and loaded")

except:
    print("No model found, rebuilding...")
    train(dataset, model, args)
    print("\nTraining complete. Saving model.")
    torch.save(model.state_dict(), MODELPATH)

print("\n", eval(model))

text = ""
while text!= "QUIT":

    text = input("\nEnter some shit to base the model off of. Might fail btw. Type QUIT to end or TRAIN to train for an additional epoch\n>")
 
    if text == "TRAIN":
        parser2 = argparse.ArgumentParser()
        parser2.add_argument('--max_epochs', type=int, default=1)
        parser2.add_argument('--batch_size', type=int, default=256)
        parser2.add_argument('--sequence_length', type=int, default=4)
        args2 = parser2.parse_args()

        print("\nTraining for one Epoch...")
        model.train()
        train(dataset, model, args2)
        print("\n", eval(model))

    else:

        try:
            words = int(input("How many words do you want to generate?"))

        #note: 'text' will be used as a key so must only contain 
        #       characters that are in the input set
            print(predict(dataset, model, text, next_words = words))

        except KeyError:
            if text=="QUIT":
                break
            else:
                print("\nBad key, try another input sequence please.")

print("\nexiting!")
