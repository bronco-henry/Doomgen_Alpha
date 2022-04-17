import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import Model
from dataset import Dataset
from train import train, predict

#benchmarking
import timeit
import time

# TODO: get rid of argparse - this is unneccessary and batch_size is not being used correctly I THINK
parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sequence_length', type=int, default=4)
args = parser.parse_args()

#benchmarking function
start_time = timeit.default_timer()
dataset = Dataset(args)
print("\n>>Loaded dataset in ", timeit.default_timer() - start_time, " seconds")

start_time = timeit.default_timer()
model = Model(dataset)
print("\n>>Model initialized in ", timeit.default_timer() - start_time, " seconds")


try: 
    start_time = timeit.default_timer()
    model.load_state_dict(torch.load(model.MODELPATH))
    model.load_epochs()
    print("DEBUGDEBUGDEBUGDEBUGDEBUGDEBUG epochs: ", model.epochs_trained)
    print("\n>>Model found and loaded in ", timeit.default_timer() - start_time, " seconds.")

except:
    print(">>No model found, rebuilding...")
    train(dataset, model, args)
    print("\n>>Training completed in ",timeit.default_timer() - start_time ," seconds. Saving model.")
    torch.save(model.state_dict(), model.MODELPATH)
    model.save_epochs()

print("\n>>", model.eval())
print(">>Epochs trained: ", model.epochs_trained)


while True:

    text = input("\n>>[optional] Enter a prompt to generate from. Type QUIT to end or TRAIN to train for an additional epoch\n>")
 
    # TODO: CLEAN UP THE LOGIC HERE ITS TOTAL SHIT
    if text == "TRAIN":
        try:
            epochs = int(input(">>How many epochs do you wish to train for?\n>"))
        except:
            epochs = 1
        parser2 = argparse.ArgumentParser()
        parser2.add_argument('--max_epochs', type=int, default=epochs)
        parser2.add_argument('--batch_size', type=int, default=256)
        parser2.add_argument('--sequence_length', type=int, default=4)
        args2 = parser2.parse_args()

        print("\n>>Training for ", epochs, " epoch(s)...")
        start_time = timeit.default_timer()
        
        # this is needed to load the model into the correct deivce for some reason
        model.train()

        train(dataset, model, args2)
        print("\n>>Training completed in ",timeit.default_timer() - start_time ," seconds. This training will not be saved.")
        print("\n", model.eval())
        print("Epochs trained: ", model.epochs_trained)

    elif text == "SAVE":
        save_choice = input("\n>>Overwrite existing model? (Y/N)\n>")
        if not save_choice or save_choice == "N" or save_choice == "n" or save_choice == "No":
            print(">>Canceling save.")
            continue

        elif save_choice.lower() == "y" or save_choice.lower() == "yes":
            print(">>Overwriting existing model.")
            torch.save(model.state_dict(), model.MODELPATH)
            model.save_epochs()

        else:
            print(">>Not understood. Canceling save.")
            continue

    elif text == "QUIT":
        break

    else:
        try:
            words = int(input(">>How many words do you want to generate?\n>"))

        # note: the string 'text' will be used as keys so must only contain 
        #       words that are in the input set.
            if words:
                start_time = timeit.default_timer()
                print("\n", "-" * 90, "\n" , "DOOM SAYS: ", "\n", predict(dataset, model, text, next_words = words), "\n", "-" * 90)
                print("\n>>Prediction completed in ",timeit.default_timer() - start_time ," seconds.")
            else:
                continue
        except KeyError:
                print("\n>>ERROR: Bad key, try another input sequence please.")
        except ValueError:
                print("\n>>ERROR: Could not interpret as a number.")

print("\n>>exiting!")
