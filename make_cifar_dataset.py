# -*- coding: utf-8 -*-
"""
Created on 20240402 
创建类似CIFAR10的训练数据集
@author: liwei
"""
import numpy as np
from PIL import Image
from os import listdir
import os
import  pickle
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from face_detector import FaceDetector

cur_file_dir = os.path.abspath(os.path.dirname(__file__))

# 初始化相关路径和文件名
flag='train'  # 指定数据集的类型，如训练集或测试集,train 或 eval
folderOriginal=cur_file_dir+"\\data\\original_{}".format(flag)  # 原始图片数据集的文件夹路径
folder32x32=cur_file_dir+"\\data\\picture32x32"  # 尺寸为32x32的图片数据集文件夹路径
binPath=cur_file_dir+"\\data\\target"  # 目标二进制文件路径

def getLabel(fname):
    """
    参数:   fname: 文件名。
    返回值: 
    label: 从文件名中提取的标签值。
    例:    getLabel("1_0.jpg")返回1
    """
    fileStr=fname.split(".")[0]
    label=int(fileStr.split("_")[0])
    return label

def getFaceImg(filePath):
    '''
        从图片中截取人脸截图
    '''
    im=Image.open(filePath)
    # 不识别人脸
    # if 1==1:
    #     return  im.resize((128,128),Image.LANCZOS)    
    
    # 等比例缩放
    # if im.size[0]/480 > im.size[1]/240:
    #     im = im.resize((480,int(480*im.size[1]/im.size[0])),Image.LANCZOS)
    # else :
    #     im = im.resize((int(240*im.size[0]/im.size[1]),240),Image.LANCZOS)
    
    # 截取人脸
    out,face_rect,landmarks_np = face_detector(im)
    # out = np.asarray(im)
    if out is None:
        return None

    # 填充至宽高一致
    width = max(out.shape[0],out.shape[1])
    x1=x2=(width-out.shape[0])//2
    if (width-out.shape[0])%2 != 0:
        x1=x1+1   
    y1=y2=(width-out.shape[1])//2
    if (width-out.shape[1])%2 != 0:
        y1=y1+1  
    out=np.pad(out, pad_width=( (x1, x2),(y1, y2),(0,0)), mode='constant', constant_values=0)
    out = Image.fromarray(out)
    #缩放至于64*64
    out = out.resize((128,128),Image.LANCZOS)      
    return out  
def img_transform(foldPath,imgList1,imgList2):
    itemsInFolder = listdir(foldPath)
    num=len(itemsInFolder)
    for i in range (0,num):
        itemName = itemsInFolder[i]
        itemPath = "{}\\{}".format(foldPath,itemName)
        if os.path.isdir(itemPath):
            img_transform(itemPath,imgList1,imgList2)    
        elif os.path.isfile(itemPath) and itemName.endswith(".jpg") :
            label=getLabel(itemName)
            #文件更名
            # if label==2:
            #      itemName=itemName.replace("2_","1_")
            #      os.rename(itemPath,"{}\\{}".format(foldPath,itemName))
            # if 1==1:
            #     continue

            out = getFaceImg(itemPath)
            if out is None:
                continue

            relativePath = "{}\\{}".format(flag,label)
            savePath = "{}\\{}".format(folder32x32,relativePath)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            saveFullPath = "{}\\{}".format(savePath,itemName)
            if os.path.exists(saveFullPath):
                os.remove(saveFullPath)
            out.save(saveFullPath)
            if label == 1:
                imgList1.append( "{}\\{}".format(relativePath,itemName)) 
            else:
                imgList2.append( "{}\\{}".format(relativePath,itemName))
def makeMyCf(imgList1,imgList2):
    data={}
    imgs1=[]
    imgs2=[]
    listFileName=[]
    #取最小的图片数量
    num=min(len(imgList1),len(imgList2))
    for k in range(0,num):
        im=Image.open("{}\\{}".format(folder32x32,imgList1[k]))
        imgs1.append(im)
        im=Image.open("{}\\{}".format(folder32x32,imgList2[k]))
        imgs2.append(im)
        print("image"+str(k+1)+"saved.")
        
    data.setdefault('imgs1'.encode('utf-8'),imgs1)
    data.setdefault('imgs2'.encode('utf-8'),imgs2)

    if not os.path.exists(binPath):
        os.makedirs(binPath)
    output = open("{}\{}.bin".format(binPath,flag), 'wb')
    pickle.dump(data, output)
    output.close()

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, L in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

def main():
    imgList1=[]
    imgList2=[]
    img_transform(folderOriginal,imgList1,imgList2)
    makeMyCf(imgList1,imgList2)

    # 求所有样本数据的均值和方差
    train_dataset = ImageFolder(root="{}\\{}".format(folder32x32,flag), transform= transforms.ToTensor())
    print("{}: {}".format('AB',getStat(train_dataset)))

if __name__ == '__main__':
    '''
    创建类似CIFAR10的训练数据集
    '''
    face_detector = FaceDetector()
    main()


    
