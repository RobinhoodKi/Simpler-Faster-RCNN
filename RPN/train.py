
import torch as t
import torch.optim as optim
import numpy  as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F

from tools.bboxtool import generate_anchor_boxes,select_anchors,label_anchors,get_index_from_array,nms
from data.dataset import  Syn_1000,Syn_100
from model.model import RegionProposalNetwork as RPN


def Draw(img,bboxs):
    '''
    img -->H,W,C
    '''
    bboxs = bboxs.astype(int)

    for j in range(len(bboxs)):
        x_min = bboxs[j][0]
        y_min = bboxs[j][1]
        x_max = bboxs[j][2]
        y_max = bboxs[j][3]

        wd =3

        img[y_min:y_min+wd,x_min:x_max,1:]=255
        img[y_max:y_max+wd,x_min:x_max,1:]=255
        img[y_min:y_max,x_min:x_min+wd,1:]=255
        img[y_min:y_max,x_max:x_max+wd,1:]=255
    return img



def test_dataset():
    dataset = Syn_1000()
    for i in range(3,4):
        img,label = dataset[i]
        H = img.shape[2]
        W = img.shape[3]
        print("H is :"+str(H))

        anchors = generate_anchor_boxes(H,W)
        print("achors' shape:"+str(anchors.shape))
        indices = select_anchors(anchors,H,W)
        print("indices' shape:"+str(len(indices)))

        IoUtable,pos_id,neg_id=label_anchors(anchors,indices,label)
        print("num of positive anchors:"+str(len(pos_id)))

        dir_picture = os.getcwd()+'/data/synthesis_1000/img_set/synthesized_image_'+str(i)+'.jpg'
        img = img/2 + 0.5
        picture = img.numpy().squeeze(0).transpose(1,2,0)


    #    picture = cv2.imread(dir_picture) # (H,W,C)
        picture =  Draw(picture,anchors[pos_id]) # Draw anchors
        #picture = Draw(picture,label)  #Draw label



        plt.imshow(picture)
        plt.show()


        #print("img Height:"+str(H)+" img Width:"+str(W))
        #print("num of positive indices are:"+str(len(pos_id)))


def test_model():
    rpn = torch.load('model.pkl',map_location='cpu')
    print("model load successfully!")
    dataset = Syn_1000()
    for i in range(135,136):
        img,label = dataset[i]
        H = img.shape[2]
        W = img.shape[3]


        anchors = generate_anchor_boxes(H,W)
        print("achors' shape:"+str(anchors.shape))

        indices = select_anchors(anchors,H,W)
        print("indices' shape:"+str(len(indices)))


        reg,cls = rpn(img)
        #prob = cls
        prob = F.softmax(cls)


        cls = cls.squeeze(0)
        reg = reg.squeeze(0)
        prob = prob.squeeze(0)


        _, predicted = torch.max(cls.data, 1)
#        print(predicted)
        predicted = predicted.numpy() # H*W*9

        pos_id = np.where(predicted==1)[0]



        pos_id = pos_id.tolist()
        print("length of pos_id is:"+str(len(pos_id)))
        pos_id = np.array(list(   set(pos_id).intersection(set(indices))))
        #pos_id = np.random.choice(pos_id,30)






        bboxes = reg[pos_id].detach().numpy()
        prob = prob.detach().numpy()




        detect = np.zeros([len(pos_id),5])  #[x1,y1,x2,y2,socrs]
        for i in range(len(pos_id)):
            detect[i,:4] = bboxes[i,:]
            detect[i,4] = prob[pos_id[i],1]
        keep = nms(detect,0.1)


        detect = detect[keep,:]
        res_id = np.where(detect[:,4]>0.7)[0]
        print(detect[:,4])
        detect = detect[res_id,:4]


        print("length after nms:"+str(detect.shape[0]))



        dir_picture = os.getcwd()+'/data/synthesis_1000/img_set/synthesized_image_'+str(i)+'.jpg'
        img = img/2 + 0.5
        picture = img.numpy().squeeze(0).transpose(1,2,0)

        picture = Draw(picture,detect)
        #picture = Draw(picture,anchors[pos_id])

        plt.imshow(picture)
        plt.show()









def train():
    ## Step1. Configure  model
    #rpn = RPN()  ## Train from the begining
    rpn = torch.load('model_6.pkl',map_location='cpu')
    ## Step2. Criterion and Optimizer
    cls_criterion = t.nn.CrossEntropyLoss()
    reg_criterion = t.nn.SmoothL1Loss() ## Not used now .
    optimizer = optim.SGD(rpn.parameters(),lr=0.001,momentum=0.9)

    ##Step3. Prepare Data
    dataset = Syn_1000()
    epchos = 1
    for j in range(epchos):
        total = 0
        correct = 0

        for i in range(len(dataset)):
            print("img:"+str(i))

            img,label = dataset[i]   ##?What's the content in the  label.
    #        print("times:"+str(i))
    #        print("img's shape:"+str(img.shape))
    #         print("label's shape:"+str(label.shape))

            H = img.shape[2]
            W = img.shape[3]
            anchors = generate_anchor_boxes(H,W)
    #        print("achors' shape:"+str(anchors.shape))
            indices = select_anchors(anchors,H,W)
    #        print("indices' shape:"+str(len(indices)))


            IoUtable,pos_id,neg_id=label_anchors(anchors,indices,label)
    #        print("num of positive indices are:"+str(len(pos_id)))

            ## Step4. Train RPN
            optimizer.zero_grad()
            reg,cls = rpn(img)
    #        print("model res come")
            cls = cls.squeeze(0)
            cls_loss = cls_criterion(cls[pos_id,:],t.ones(len(pos_id),dtype=t.long)) + cls_criterion(cls[neg_id,:],t.zeros(len(neg_id),dtype=t.long))
            print("cls loss:"+str(cls_loss))

            ####***reg_loss**####
            reg = reg.squeeze(0)
            ## Prediction
            reg = reg[pos_id,:]  # (N,4)






        #    print("reg shape is:"+str(reg.shape))

            # labels
            pos_id_iou = get_index_from_array(indices,pos_id)
                # ndarray-->Tensor
            pos_id_iou = torch.from_numpy(pos_id_iou)
            IoUtable = torch.from_numpy(IoUtable)
            label = torch.from_numpy(label).float()



        #    print("pos_id_iou length is:"+str(len(pos_id_iou)))
            _,labels_id = torch.max(IoUtable[pos_id_iou],dim=1)
            label = label[labels_id,:] # (N,4)


        #    print("label shape is:"+str(label.shape))

            # anchors
            anchor = anchors[pos_id,:]  # (N,4)
               ##ndarray-->Tensor
            anchor = torch.from_numpy(anchor).float()


            w_a = anchor[:,2]-anchor[:,0]
            h_a = anchor[:,3]-anchor[:,1]


        #    print("anchor shape is:"+str(anchor.shape))

            # anchors &  labels
            t_x_l = (label[:,0]-anchor[:,0])/w_a
            t_y_l = (label[:,1]-anchor[:,1])/h_a
            t_x2_l = (label[:,2]-anchor[:,2])/w_a
            t_y2_l = (label[:,3]-anchor[:,3])/h_a

            # anchors & predictions
            t_x = (reg[:,0]-anchor[:,0])/w_a
            t_y = (reg[:,1]-anchor[:,1])/h_a
            t_x2 = (reg[:,2]-anchor[:,2])/w_a
            t_y2 = (reg[:,3]-anchor[:,3])/h_a



        #    print("reg_loss 1:"+str(reg_criterion(t_x_l,t_x)))
        #    print("reg_loss 2:"+str(reg_criterion(t_y_l,t_y)))
        #    print("reg_loss 3:"+str(reg_criterion(t_x2_l,t_x2)))
        #    print("reg_loss 4:"+str(reg_criterion(t_y2_l,t_y2)))





            reg_loss = reg_criterion(t_x_l,t_x) + reg_criterion(t_y_l,t_y) +  reg_criterion(t_x2_l,t_x2) +  reg_criterion(t_y2_l,t_y2)
            print("reg loss:"+str(reg_loss))
            ####***reg_loss**####

            loss = 0.1*reg_loss + cls_loss
            print("total loss "+str(i)+" :"+str(loss))
            loss.backward()


            optimizer.step()
            #print("times:"+str(j)+",i is "+str(i)+",loss:"+str(cls_loss))
            ## Step5. Validate during the training.

#            if (i % 20 == 0):
#                index = np.random.randint(0,len(dataset))
#                img,object,label = dataset[index]
#                reg,cls = rpn(img)
#                cls = cls.squeeze(0)
#                _,predicted = t.max(cls,1)
#                H = img.shape[2]
#                W = img.shape[3]
#                anchors = bbox.generate_anchor_boxes(H,W)
#                indices = select_anchors(anchors,H,W)
#                pos_id,neg_id=label_anchors(anchors,indices,label)
#                total  += len(pos_id)+len(neg_id)
#                for i in range(len(pos_id)):
#                    if (predicted[pos_id[i]]==1):
#                        correct +=1
#                for i in range(len(neg_id)):
#                    if (predicted[neg_id[i]]==0):
#                        correct +=1
#                print("***Testing***:total: "+str(total)+" correct: "+str(correct))
#                print("Accuracy is : "+str(correct/(2*len(pos_id))))



test_model()
#test_dataset()
#test2()
#train()
