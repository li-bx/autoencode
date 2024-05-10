import os
import sys

cur_file_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_file_dir)

import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import sys
import time
from PIL import Image
import torch.nn as nn
from deepfake import Autoencoder
from face_detector import FaceDetector
import torchvision.transforms as transforms

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def correct_colors(im1, im2, landmarks1,COLOUR_CORRECT_BLUR_FRAC = 0.6):
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0,0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0,0)
    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /im2_blur.astype(np.float64))

def change_face(im):
    # 加载图片
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 等比例缩放
    if im.size[0]/640 > im.size[1]/480:
        im = im.resize((640,int(640*im.size[1]/im.size[0])),Image.LANCZOS)
    else :
        im = im.resize((int(480*im.size[0]/im.size[1]),480),Image.LANCZOS)
    
    # 截取人脸
    out,face_rect,landmarks = face_detector(im)
    # out = np.asarray(im)
    if out is None:
        return None
    
    # 填充至宽高一致
    original_shape = out.shape
    if original_shape[0] == 0:
        return None
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
    
    out = transforms.ToTensor()(out)
    train_normalize = transforms.Normalize([0.4322424, 0.34108055, 0.29687616], [0.33795825, 0.28329322, 0.2550639])
    out = train_normalize(out)

    # out =torch.tensor(np.array(out,np.float32)).to(device)
    out = out.reshape(1,3,128,128).to(device)
    
    with torch.no_grad():
        out = autoencoder(out,'B')[1].reshape(3,128,128).permute(1,2,0)

    out = out.cpu().numpy() 
    out = Image.fromarray((out*255).astype(np.uint8))

    image_array = np.asarray(im)
    size = (image_array.shape[0], image_array.shape[1])
    mask = face_detector.get_face_mask(size, landmarks)  # 脸图人脸掩模
    mask = np.stack((mask,mask,mask),axis=2)
    image_face = np.multiply(image_array,(1-mask)).astype('uint8')


    # out = np.multiply(image_array,mask)[face_rect.top():face_rect.bottom()+1,face_rect.left():face_rect.right()+1]
    # # 还原原始尺寸
    out = out.resize((width,width),Image.LANCZOS)
    out = np.array(out)[x1:width-x2,y1:width-y2,:]
    out = cv2.drawMarker(out,(50,10),(0,0,255),markerType=cv2.MARKER_CROSS,markerSize=10,thickness=2)
    # out=np.pad(out, pad_width=( (face_rect.top(),image_array.shape[0]-face_rect.bottom()),(face_rect.left(), image_array.shape[1]-face_rect.right()),(0,0)), mode='constant', constant_values=0)
    out=np.pad(out, pad_width=( (landmarks[:,1].min(),image_array.shape[0]-landmarks[:,1].max()-1),(landmarks[:,0].min(), image_array.shape[1]-landmarks[:,0].max()-1),(0,0)), mode='constant', constant_values=0)
    strength:int = 50
    out = Image.fromarray(out.astype('uint8')).resize((image_array.shape[1]+strength,image_array.shape[0]+strength),Image.LANCZOS)
    # mask = Image.fromarray(mask.astype('uint8'))
    # mask = np.pad(out_correct,((-3,3),(3,3),(0,0)),'constant',constant_values=(0,0))
    offset:int = strength//2
    out = np.array(out)[offset:-offset,offset:-offset,:]

    out=np.multiply(out,mask)
    out=out + (mask - 1)

    out_correct =  correct_colors(image_array,out,landmarks,0.6)
    out_correct = Image.fromarray(out_correct.astype('uint8'))

    r = image_face +  np.multiply(out_correct,mask)
    return r

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, oriImg = cap.read()
        canvas = copy.deepcopy(oriImg)
        canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        canvas = change_face(canvas)
        if canvas is None:            
            cv2.imshow('demo', oriImg)
        else:
            canvas = Image.fromarray(canvas.astype(np.uint8))
            # canvas = canvas.resize((640,480),Image.LANCZOS)
            cv2.imshow('demo',cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR)) 

        # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        # canvas = cv2.resize(canvas, (oriImg.shape[1], oriImg.shape[0]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 加载模型
    FLAG = 1
    WEIGHT_PATH=os.path.join(cur_file_dir,'weights\epoch_weight%d.pth' % (FLAG))
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(WEIGHT_PATH))
    autoencoder.eval()     
    face_detector = FaceDetector()
    main()