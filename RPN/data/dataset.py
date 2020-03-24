import os
from PIL import Image
import numpy as np
import torch as t
from torch.utils import data
import json
import torchvision.transforms as T
import glob
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([.5,.5,.5],[.5,.5,.5])
])


class Syn_1000(data.Dataset):
    def __init__(self):
        root_img = os.getcwd()+'/data/synthesis_1000/img_set'
        imgs = os.listdir(root_img)
        self.imgs = [os.path.join(root_img,img) for img in imgs]
        self.imgs.sort()

        ## label
        f = open(os.getcwd()+'/data/synthesis_1000/data_indicator.json')
        self.labels = json.load(f)

        f.close()



    def __getitem__(self,index):
        pil_img = Image.open(self.imgs[index])
        w, h = pil_img.size
        pil_img.thumbnail((w//2, h//2))
        data = transform(pil_img).unsqueeze(0)
        num_objects=len(self.labels[index]['objects'])
        bbox = np.zeros([num_objects,4])
        for i in range(num_objects):
            bbox[i,:] = np.array(self.labels[index]['objects'][i]['bbox'])
        bbox[:,2] = bbox[:,2]+bbox[:,0] # (x1,y1,w,h)  --> (x1,y1,x2,y2)
        bbox[:,3] = bbox[:,3]+bbox[:,1]

        bbox[:,:] = bbox[:,:]//2

        return data,bbox




    def __len__(self):
        return len(self.imgs)
class Syn_100(data.Dataset):
    def __init__(self):
        root_img = os.getcwd()+'/data/synthesis_100'
        imgs = os.listdir(root_img)
        self.imgs = [os.path.join(root_img,img) for img in imgs]
        self.imgs.sort()

        ## label
        f = open(os.getcwd()+'/data/synthesis_1000/data_indicator.json')
        self.labels = json.load(f)

        f.close()



    def __getitem__(self,index):
        pil_img = Image.open(self.imgs[index])
        w, h = pil_img.size
        pil_img.thumbnail((w//2, h//2))
        data = transform(pil_img).unsqueeze(0)
        num_objects=len(self.labels[index]['objects'])
        bbox = np.zeros([num_objects,4])
        for i in range(num_objects):
            bbox[i,:] = np.array(self.labels[index]['objects'][i]['bbox'])
        bbox[:,2] = bbox[:,2]+bbox[:,0] # (x1,y1,w,h)  --> (x1,y1,x2,y2)
        bbox[:,3] = bbox[:,3]+bbox[:,1]

        bbox[:,:] = bbox[:,:]//2

        return data,bbox




    def __len__(self):
        return len(self.imgs)
