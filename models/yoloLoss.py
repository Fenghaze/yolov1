# -*- coding: utf-8 -*-
"""
@Time          : 2020/08/12 18:30
@Author        : FelixFu / Bryce
@File          : yoloLoss.py
@Noice         :
@Modificattion :
    @Detail    : a little dufficult in builting yoloLoss funcion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S  # 网格大小
        self.B = B  # 预测的bbox个数
        self.l_coord = l_coord  # 超参数λcoord=5
        self.l_noobj = l_noobj  # 超参数λnoobj=0.5

    def compute_iou(self, box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """
        # 首先计算两个box左上角点坐标的最大值和右下角坐标的最小值，然后计算交集面积，最后把交集面积除以对应的并集面积
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(  # 左上角的点
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(  # 右下角的点
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 指两个box没有重叠区域
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        """
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        """
        N = pred_tensor.size()[0]   # batchsize
        # 具有目标标签的索引(bs, 14, 14, 30)中14*14方格中的哪个方格包含目标
        # target_tensor[:, :, :, 4]是置信度值，如果置信度>0，说明有目标
        coo_mask = target_tensor[:, :, :, 4] > 0  # coo_mask.shape = [bs, 14, 14]
        noo_mask = target_tensor[:, :, :, 4] == 0  # 不具有目标的标签索引


        # 得到含物体的坐标等信息(coo_mask扩充到与target_tensor一样形状, 沿最后一维扩充)
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)  # 有目标的位置为True，大部分为False
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)  # 有目标的位置为False，大部分为True

        # coo_pred 取出预测结果中有物体的网格，并改变形状为 [xxx,30]
        # xxx代表一个batch的图片上的存在物体的网格总数
        coo_pred = pred_tensor[coo_mask].view(-1, 30)
        # box_pred 2个bbox的预测值
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1], [x2,y2,w2,h2,c2]
        # class_pred 网格的分类概率
        class_pred = coo_pred[:, 10:]

        # 真实标签中有物体的网格对应的数值
        coo_target = target_tensor[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # ---------计算损失1：计算不包含目标，即标签为0的网格的损失------------
        noo_pred = pred_tensor[noo_mask].view(-1, 30)   # 如tensor([195, 30])，表示有195个没有目标的网格，值为预测值
        noo_target = target_tensor[noo_mask].view(-1, 30)   # 标签值，均为0

        # 定义一个与noo_pred同等shape的掩码
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()).bool()
        noo_pred_mask.zero_()   # 初始化为0
        # bbox置信度置为True
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        # 使用掩码获取noo_pred对应位置的置信度值
        noo_pred_c = noo_pred[noo_pred_mask]  # tensor([390])，390=195*2，表示获取了195个网格中的所有预测的置信度值
        noo_target_c = noo_target[noo_pred_mask]    # 所有真实标签的置信度值，均为0
        # -----公式（4）：网格没有物体时的置信度损失-----
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)

        # ---------计算损失2：计算包含目标的网格的损失---------------------------
        # 定义一个与box_target同等shape的掩码
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()
        coo_not_response_mask.zero_()
        # iou
        box_target_iou = torch.zeros(box_target.size()).cuda()
        # 从有目标的网格中，选择最好的IOU
        for i in range(0, box_target.size()[0], 2):
            box1 = box_pred[i:i+2]   # 第i个有目标的网格的2个bbox
            box1_xyxy = torch.FloatTensor(box1.size())
            # (x,y,w,h)
            box1_xyxy[:, :2] = box1[:, :2]/14. - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2]/14. + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)    # 第i个有目标的网格的bbox标签
            box2_xyxy = torch.FloatTensor(box2.size())
            box2_xyxy[:, :2] = box2[:, :2]/14. - 0.5*box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2]/14. + 0.5*box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0) # 获取最大的iou值及其bbox索引
            max_index = max_index.data.cuda()
            # 设置掩码，将索引位置对应的bbox设为true
            coo_response_mask[i+max_index] = 1
            coo_not_response_mask[i+1-max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            # iou value 作为box包含目标的confidence(赋值在向量的第五个位置)
            box_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = box_target_iou.cuda() # tensor[[0,0,0,0,0], [0,0,0,0,max_iou)]]

        # 使用coo_response_mask选择最佳的预测bbox和对应的iou
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        # ----公式（3）：网格有物体时的置信度损失----
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        # ----公式（1）+（2）：网格有物体时的bbox位置损失-----
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)

        # 2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        # I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # -----公式（5）：类别损失-------
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        return (self.l_coord*loc_loss + self.B*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N




