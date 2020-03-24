
import numpy as np


###########----> anchor size tool <--------#########

def generate_anchor_box(subsample=16,scales=[4,8,16],ratio=[0.5,1,2]):
    anchor = np.zeros((9,4),dtype=np.float32)
    center_x = subsample/2
    center_y = subsample/2
    for i in range(3):
        for j in range(3):
            H = subsample*scales[i]*np.sqrt(ratio[j])
            W = subsample*scales[i]*np.sqrt(1/ratio[j])
            index = i*3+j
            # [x1,y1,x2,y2 ]
            anchor[index,]=[center_x-W/2,center_y-H/2,center_x+W/2,center_y+H/2]
    anchor = anchor.astype(int)
    return anchor





def generate_anchor_boxes(img_H,img_W,subsample=16):
    num = (img_H//subsample) * (img_W//subsample)
    anchors = np.zeros((num,9,4),dtype=np.int32)
    anchor = generate_anchor_box(subsample=subsample)
    for i in range(img_H//subsample):
        for j in range(img_W//subsample):
            index = i*(img_W//subsample) + j
            anchors[index,:,0:3:2] = anchor[:,0:3:2]+subsample*j
            anchors[index,:,1:4:2] = anchor[:,1:4:2]+subsample*i
    anchors = anchors.reshape(-1,4)
    return anchors


###########----> anchor size tool <--------#########



def get_index_from_array(from_array, purpose_array):
    purpose_array = np.array(purpose_array)
    from_array = np.array(from_array)
    purpose_idx_in_from = -np.ones(purpose_array.shape).astype(int)     #初始化待返回的索引数组为-1，长度对应purpose_array
    p_idx = np.in1d(purpose_array, from_array)      #筛选出 purpose_array 存在于 from_array 中存在的项
    union_array = np.hstack((from_array, purpose_array[p_idx]))         #合并from_array 和从 purpose_array 中筛选出来的数组
    _, union_idx, union_inv = np.unique(union_array, return_index=True, return_inverse=True)    #unique函数得到索引值
    purpose_idx_in_from[p_idx] = union_idx[union_inv[len(from_array):]] #待返回的索引数组赋值
    return purpose_idx_in_from



def select_anchors(anchors,H,W):
    indices = []
    for i in range(anchors.shape[0]):
        # [x1,y1,x2,y2]
        if(anchors[i,0]>=0 and anchors[i,0]<W and anchors[i,2]>=0 and anchors[i,2]<W):
            if(anchors[i,1]>=0 and anchors[i,1]<H and anchors[i,3]>=0 and anchors[i,3]<H):
                indices.append(i)
    return indices




def IoU( bbox_a, bbox_b):
    """


        Args:
            bbox_a (array): An array whose shape is :math:`(N, 4)`.
                :math:`N` is the number of bounding boxes.
                The dtype should be :obj:`numpy.float32`.
            bbox_b (array): An array similar to :obj:`bbox_a`,
                whose shape is :math:`(K, 4)`.
                The dtype should be :obj:`numpy.float32`.

        Returns:
            array:
            An array whose shape is :math:`(N, K)`. \
            An element at index :math:`(n, k)` contains IoUs between \
            :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
            box in :obj:`bbox_b`.

    """

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)











def label_anchors(anchors,indices,labels):
    pos_id = []
    neg_id = []

    IoU_table =  IoU(anchors[indices],labels)
    for i in range(IoU_table.shape[0]):
        if max(IoU_table[i,:])>0.53 : #0.53
            pos_id.append(indices[i])
        elif max(IoU_table[i,:])<0.3:
            neg_id.append(indices[i])
    # Only Select at most 128 anchors to train
    if len(pos_id)==0:
        pos_id.append(indices[np.argmax(IoU_table)//len(labels)])
    num = min(128,min(len(pos_id),len(neg_id)))

    pos_id = np.sort(np.random.choice(pos_id,num,replace=False))
    neg_id = np.sort(np.random.choice(neg_id,num,replace=False))
    return IoU_table,pos_id,neg_id










def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 每个boundingbox的面积
    order = scores.argsort()[::-1] # boundingbox的置信度排序
    keep = [] # 用来保存最后留下来的boundingbox
    while order.size > 0:
        i = order[0] # 置信度最高的boundingbox的index
        keep.append(i) # 添加本次置信度最高的boundingbox的index

        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    keep = np.array(keep)

    return keep
