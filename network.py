import torch
import torchvision.models as models
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def createResnet50(input_num, output_num):
    model = models.resnet50(pretrained=True)

    model.fc = nn.Sequential(
            nn.Linear(
                in_features=input_num,
                out_features=output_num
            ),

            nn.Sigmoid()
        )

    return model.to(device)


class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.resnet50 = createResnet50(2048, 2)
        

    def forward(self, x1, x2, x3, resnet=True):
        output1 = self.resnet50(x1)
        output2 = self.resnet50(x2)
        output3 = self.resnet50(x3)
        
        return output1, output2, output3

    def get_embedding(self, x, resnet=True):
        return self.resnet50(x)