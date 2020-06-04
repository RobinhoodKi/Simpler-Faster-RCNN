# Simpler-Faster-RCNN  (in Pytorch)
Overview: Implement and train faster-RCNN by hand.

Goal: Object detection on RPC [(A Large-Scale Retail Product Checkout Dataset)](https://www.kaggle.com/diyer22/introduce-rpc-dataset).

## Introduction
In order to make the procedure clear,I decide to implement and train RPN and fast-RCNN seperately.Because the main inovation part of faster-RCNN is the RPN architecture, so I implement RPN at first.


## Region Proposal Network
To evaluate my model performance, you can direcly execute:
```
python train.py
```
training set

|classfication(anchors) | regression |
|----------|--------|
|<img src="https://github.com/RobinhoodKi/Simpler-Faster-RCNN/blob/master/RPN/result/cls_v.png" width="800">|<img src="https://github.com/RobinhoodKi/Simpler-Faster-RCNN/blob/master/RPN/result/reg_v.png" width="800">|

testing set

|test_1 | test_2 |  test_3|
|----------|--------|------|
|<img src="https://github.com/RobinhoodKi/Simpler-Faster-RCNN/blob/master/RPN/result/t_1.png" width="800">|<img src="https://github.com/RobinhoodKi/Simpler-Faster-RCNN/blob/master/RPN/result/t_2.png" width="800">|<img src="https://github.com/RobinhoodKi/Simpler-Faster-RCNN/blob/master/RPN/result/t_3.png" width="800">|

## Fast-RCNN
Building now..

## Acknowledgement
Two important functions' implementation, *IoU* and *nms* ,are heavily influenced by [*chenyuntc*'s code](https://github.com/chenyuntc/simple-faster-rcnn-pytorch).

Especially the *IoU* function, *chenyuntc*'s code is 50 times faster than my original implementation.

My original implementation used **for loop** to calculate the IoU table between anchors and labels,It took 10 seconds for 1 img, which makes it too slow to train network.
