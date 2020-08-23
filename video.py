# -- coding: utf-8 --
# @Time : 2020/8/21 21:54
# @Author : zhl
# @File : video.py.py
# @Desc: 

import torch

from models.resnet_yolo import resnet50
import cv2
import numpy as np
import argparse
import time
from predict import predict_camera, Color, VOC_CLASSES
use_cuda = True

def plot_boxes_cv2(model, image_name):
    image = np.copy(image_name)
    print('predicting...')
    result = predict_camera(model, image_name)

    for left_up, right_bottom, class_name, _, prob in result:
        print("class_name:%s, prob:%f" % (class_name, prob))
        if prob >= 0.5:
            color = Color[VOC_CLASSES.index(class_name)]
            cv2.rectangle(image, left_up, right_bottom, color, 2)
            label = class_name + str(round(prob, 2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                          color, -1)
            cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
    return image

def detect_cv2_camera(weightfile):
    import cv2
    model = resnet50()
    model.load_state_dict(torch.load(weightfile))
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        model.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    while True:
        ret, img = cap.read()   # 获取当前帧
        start = time.time()
        result_img = plot_boxes_cv2(model, img)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        cv2.imshow('Yolov1 demo', result_img)
        cv2.waitKey(1)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-weightfile', type=str,
                        default='best.pth',
                        help='path of trained model.', dest='weightfile')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    detect_cv2_camera(args.weightfile)
