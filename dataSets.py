import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class TripletCIFARTrain(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = dataset.transform

        self.labels = self.dataset.targets
        self.labels = np.array(self.labels)
        self.data = self.dataset.data
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}
                                        
    
    def __getitem__(self, index): 
        img1, label1 = self.data[index], self.labels[index]
        
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])

        negative_label = np.random.choice(list(self.labels_set - set([label1])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        
        img2 = self.data[positive_index]
        img3 = self.data[negative_index]
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        
        return img1, img2, img3
    
    def __len__(self):
        return len(self.dataset)
