import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.manifold import TSNE
import pandas as pd

from utils import get_all_features, plot_features_tsne
from network import TripletNet
from mahalanobisClassifier import MahalanobisClassifier
from tripletLoss import TripletLoss

def train_epoch(model, dataLoader, loss_fn, optimizer, lr=0.0001, device='cuda'):
    model.train() #Tells to torch that we are training de model

    losses = {
        'losses_history': [],
        'total_loss': 0.0
    }

    for i, data in enumerate(dataLoader):

        image1, image2, image3 = data

        image1 = image1.view(-1, 3, 32, 32).to(device)
        image2 = image2.view(-1, 3, 32, 32).to(device)
        image3 = image3.view(-1, 3, 32, 32).to(device)

        optimizer.zero_grad()
        outputs = model(image1, image2, image3)

        loss = loss_fn(*outputs)
        losses['losses_history'].append(loss)
        losses['total_loss'] += loss.item()
        loss.backward()
        optimizer.step()

    losses['total_loss'] /= (i + 1) 

    return losses
        

def train(model, dataLoader, device, PATH):
    print('Starting Training...')
    
    model.train()
    
    margin = 1
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_epochs = 50

    min_loss = 1000.0

    for epoch in range(n_epochs):

        print(f'\tEpoch: {epoch+1}/{n_epochs}')
        
        losses = train_epoch(model, dataLoader, loss_fn, optimizer, lr, device)
        loss = losses['total_loss']
        print(f'\t\tLoss in this epoch: {loss}')

        if min_loss > loss:
            min_loss = loss

            #torch.save(model.state_dict(), PATH)
            print('\t\tModel Save!!')

    print('Finished training')


def validation(train_dataLoader, test_dataLoaer, device, PATH):
    print('Starting Validation...')

    model = TripletNet()
    model.load_state_dict(torch.load(PATH))

    #Features extraction
    train_features, train_labels = get_all_features(train_dataLoader, model)
    test_features, test_labels = get_all_features(test_dataLoaer, model)

    pd_train_features = pd.DataFrame(train_features)
    pd_test_features = pd.DataFrame(test_features)
    
    #Mahalanobis classfier
    clf = MahalanobisClassifier(pd_train_features, train_labels)
    pred_probs = clf.predict_probability(pd_test_features)
    labelsMahalanobis = clf.predict_class(pd_test_features, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    '''
    accuracy = accuracy_score(test_labels, labelsMahalanobis) * 100
    adjusted = adjusted_rand_score(test_labels, labelsMahalanobis) * 100
    print(f'Accuracy: {accuracy}%')
    print(f'Adjusted Rand Score: {adjusted}%')
    '''

    #TSNE 
    tsne_train = TSNE(n_components=2).fit_transform(train_features)
    tsne_test = TSNE(n_components=2).fit_transform(test_features)
    
    plot_features_tsne(tsne_train, train_labels, name_save='train-features-tsne')
    plot_features_tsne(tsne_test, test_labels, name_save='test-features-tsne')
    
    print('Validation completed!')

