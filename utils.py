import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from network import TripletNet

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from dataSets import TripletCIFARTrain


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def createTripletDataLoaders(train_data_set, test_data_set):
    config = {
        'batch_size': 128,
        'num_workers': 2
    }

    triplet_train_data_set = TripletCIFARTrain(train_data_set)
   

    triplet_train_data_loader = DataLoader(triplet_train_data_set, **config)
    triplet_test_data_loader = DataLoader(test_data_set, **config)

    return triplet_train_data_loader, triplet_test_data_loader


def get_features_vector(model, image):
    image = image.unsqueeze(0)

    # Use the model object to select the desired layer
    layer = model.resnet50._modules.get('avgpool')

    # Set model to evaluation mode
    model.eval()

    # 1. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 2048
    my_embedding = torch.zeros(2048)

    # 2. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    
    # 3. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    
    # 4. Run the model on our transformed image
    model.get_embedding(image)
    
    # 5. Detach our copy function from the layer
    h.remove()
    
    # 6. Return the feature vector
    return my_embedding.numpy()


def get_all_features(dataLoader, model):
    outputs = []
    labels = []

    for images, targets in dataLoader:
        
        labels.extend(targets.cpu().data.numpy())
        
        images = images.view(-1, 3, 32, 32).to(device)
        
        for image in images:

            output = get_features_vector(model, image)
            outputs.append(output)

    labels = np.array(labels)
    outputs = np.array(outputs)

    return outputs, labels

def plot_features_tsne(features, targets, name_save='features', xlim=None, ylim=None):
    cifar_classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 
                        'Frog', 'Horse', 'Ship', 'Truck']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf']

    plt.figure(figsize=(10, 10))

    for i in range(10):

        inds = np.where(targets == i)[0]
        plt.scatter(features[inds, 0], features[inds, 1], alpha=0.5, color=colors[i])

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    
    plt.legend(cifar_classes)
    plt.savefig(name_save, format='png')