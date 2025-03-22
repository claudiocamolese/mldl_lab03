import torch.nn as nn
import torch
import torchvision
import torchvision.transformer as F

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet,self).__init__()
        self.cv1 = nn.Conv2d(3,64, kernel_size=3, padding=1, stride=2)
        self.cv2= nn.Conv2d(64,128, kernel_size=3,padding=1,stride=2)
        self.cv3= nn.Conv2d(128,256, kernel_size=3, padding=1, stride=2)
        self.cv4= nn.Conv2d(256,512, kernel_size=3, padding=1, stride=2)
        self.cv5= nn.Conv2d(512,1024, kernel_size=3, padding=1, stride=2)
        
        self.flatten= nn.Flatten(2)
        
        self.fc1= nn.Linear(1024,512)
        self.fc2= nn.Linear(512,200)
        
    def forward(self,x):
        x= ...