
from dataset import Syn_1000
import json
import os

def test1():
    dataset = Syn_1000()
    for i in range(len(dataset)):
        img , bbox = dataset[i]
        print(img.shape)






def test2():
    f =  open(os.getcwd()+'/synthesis_1000/data_indicator.json')
    data = json.load(f)
    f.close()
    for i in range(5):
        print(data[i]['image_id'])

test1()
