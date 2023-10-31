from torchvision import datasets, transforms
from PIL import Image
import torch
import json
import sys

def get_categorymap(jsonfile):
    try:
        with open('cat_to_name.json', 'r') as f:
            return json.load(f)
    except Exception as error:
        print('\n***Error with JSON category map file:', type(error).__name__, '-', error, '***\n')
        sys.exit()

def load_trainimages(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    try:    
        train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) 
    except Exception as error:
        print('\n***Error with checkpoint file:', type(error).__name__, '-', error, '***\n')
        sys.exit()
    else:
        return trainloader, train_data

def load_testimages(test_dir, load_images = True):
    """
    If load_images is True test_dir, should be a directory to a set of images to test a model.
    If load_images is False, test_dir, should be a path to an image so that it can be processed for
    input into the model"""
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    try:
        if load_images:
            test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
            testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
            return testloader
        else:
            with Image.open(test_dir) as im:
                return test_transforms(im)
    except Exception as error:
        if load_images:
            print('\n***Error with Test and Validation Data:', type(error).__name__, '-', error, '***\n')
            print('\n***Test and Validation data might not have the right format and arrangement***\n')
            sys.exit()
        else:
            print('\n***Error with Image File:', type(error).__name__, '-', error, '***\n')
            print('\n***Image file might be corrupted or not in the right format***\n')
            sys.exit()
        




