import argparse
import sys
import torch
import os.path
from tabulate import tabulate

def input_arg():
    """
    This function collects input argument from the command line for train.py script
    """
    parser = argparse.ArgumentParser(description = 'Image classifier program')
    parser.add_argument('data_dir', type = str, help = 'Directory to the folder containing the data')
    
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Input model of type vgg, alexnet, densenet and resnet')
    parser.add_argument('--learning_rate', type = float, default = 0.003, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, default = 5, help = 'Input training epoch')
    parser.add_argument('--gpu', action = 'store_const', const = 'cuda', default = 'cpu', help = 'Choose gpu for training')
    parser.add_argument('--hidden_units', type = int, help = 'number of hidden units', default = [512], nargs = '+')
    parser.add_argument('--save_dir', type = str, help = 'Directory to save checkpoint', default = 'checkpoint.pth')
    parser.add_argument('--output', type = int, help = 'output of classifier', default = 102)
    
    return parser.parse_args()

def predict_args():
    """
    This function collects input argument from the commandline for predict.py script
    """
    parser = argparse.ArgumentParser(description = 'Image classifier program')
    
    parser.add_argument('image', type = str, help = 'Image path')
    parser.add_argument('checkpoint', type = str, help = 'Checkpoint path')
    parser.add_argument('--top_k', type = int, help = 'Top k most likely classes', default = 5)
    parser.add_argument('--category_names', type = str, help = 'path to map of categories', default = 'cat_to_name.json')
    parser.add_argument('--gpu', action = 'store_const', const = 'cuda', default = 'cpu', help = 'choose gpu for training')
    
    return parser.parse_args()

def checkgpu(user_input):
    """
    If the user intends to train on the GPU, this function verifies the availability of a GPU and exits the code if none is found.
    """
    if user_input == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        return device
    elif(user_input == 'cpu'):
        device = torch.device('cpu')
        return device
    else:
        print('\n***GPU is not available***\n')
        sys.exit()
        
def has_file(path):
    """
    This function checks if a file exists, and if it doesn't exit the code to prevent program crashing
    """
    if not os.path.isfile(path):
        name = os.path.basename(os.path.normpath(path))
        print("\n***This file: " + name + " does not exist***\n")
        sys.exit()
        
def has_dir(directory, save = True):
    """
    This function checks if a directory exists, and if it doesn't exit the code to prevent program crashing
    """
    if save:
        head = os.path.split(os.path.normpath(directory))
        if head[0] == '':
            return
        elif not os.path.isdir(head[0]):
            print("\n***The directory: '" + head[0] +"' does not exist, therefore checkpoint wont save***\n")
            sys.exit()
    elif not os.path.isdir(directory):
        print("\n***The directory: '" + directory +"' does not exist***\n")
        sys.exit()

        
       
def print_data(prob, predict):
    """
    This function prints inference result in a table format
    """
    header = ['Probability', 'Classes']
    data = [[round(pr, 3), p] for pr, p in zip(prob, predict)]
    print(tabulate(data, header))
    

