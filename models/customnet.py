import torch.nn as nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.cv1= nn.Conv2d(3, 6, 5) #6 kernel di dimensione 5x5. Input size (3x32x32)
        self.cv2= nn.Conv2d(6, 16, 5)
        self.fc1= nn.Linear(16*5*5,120)
        self.fc2= nn.Linear(120,84)
        self.fc3= nn.Linear(84,10)

    def forward(self, x):
        x= F.relu(self.cv1(x)) #dimensione di output (Input-kernel)/stride +1 = (32-5)/1 +1=28 --> (6,28,28)
        x= F.max_pool2d(x, (2,2)) #maxpool2d --> (6,14,14) 
        x= F.relu(self.cv2(x)) # dimensione di output (14-5)/1 +1=10 --> (16,10,10)
        x= F.max_pool2d(x, (2,2)) #maxpool2d --> (16, 5, 5)  ----> motivo per cui fc1 ha quei valori di input
        x= x.view(-1, self.flattened_features(x))
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
        return x

    def flattened_features(self, x):
        size = x.size()[1:]
        num_feats=1
        for s in size:
            num_feats *= s
        return num_feats