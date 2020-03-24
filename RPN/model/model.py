import torch
import torch.nn as nn
from torchvision import models


class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super(RegionProposalNetwork,self).__init__()
        VGG = models.vgg16_bn(pretrained=True)
        self.extractor = VGG.features[:-1]
        self.mid = nn.Conv2d(512,512,3,stride=1,padding=1)
        self.cls = nn.Conv2d(512,2*9,1,stride=1,padding=0)
        self.reg = nn.Conv2d(512,4*9,1,stride=1,padding=0)



    def forward(self,x):
        x = self.extractor(x)
        x = self.mid(x)
        # Suppose One img at a time.
        reg = self.reg(x).permute(0,2,3,1).reshape(1,-1,4)
        cls = self.cls(x).permute(0,2,3,1).reshape(1,-1,2)


        return reg,cls






'''
rpn = RegionProposalNetwork()
data = torch.ones(1,3,800,800)
reg,cls = rpn(data)

print(reg.shape)
print(cls.shape)
'''
