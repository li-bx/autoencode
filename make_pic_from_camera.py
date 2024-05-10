import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import os
from face_detector import FaceDetector


plt.switch_backend('agg')

pwd_path = os.path.abspath(os.path.dirname(__file__))
# 初始化相关路径和文件名
flag='train'  # 指定数据集的类型，如训练集或测试集,train或eval
person='B'     # 指定采集图片的人名，A or B
start_index=1  # 指定从第几张图片开始采集
folderOriginal=pwd_path+"\\data\\original_%s" % (flag)  # 原始图片数据集的文件夹路径

if __name__ == "__main__":
    '''
    使用摄像头采集图片，并保存到指定文件夹
    '''
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    face_detector = FaceDetector()
    to_save = False
    count=start_index
    while True:
        ret, oriImg = cap.read()
        canvas = copy.deepcopy(oriImg)
        cv2.imshow('camera', canvas)   

        if to_save and face_detector.have_one_face(image=oriImg):
            cv2.imwrite(os.path.join(folderOriginal, '%d_%06d.jpg' % (ord(person) - ord('A') + 1, count)), oriImg)
            count+=1
            to_save=False 
            print('save image:', count)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            to_save = True

    cap.release()
    cv2.destroyAllWindows()

