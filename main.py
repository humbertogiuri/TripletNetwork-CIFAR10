from utils import createTripletDataLoaders
from network import TripletNet
from train import train, validation

import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = './resnet.pth'

train_data_set = CIFAR10('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    

test_data_set = CIFAR10('./data', train=False, download=True,
                                    transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

triplet_train_data_loader, test_data_loader = createTripletDataLoaders(train_data_set, test_data_set)

model = TripletNet()

argumento = input('Type train for Train or val for Validation: ')


if(argumento == 'train'):
    train(model, triplet_train_data_loader, device, PATH)

elif(argumento == 'val'):
    config = {
        'batch_size': 128,
        'num_workers': 2
    }

    train_data_loader = DataLoader(train_data_set, **config)

    validation(train_data_loader, test_data_loader, device, PATH)

