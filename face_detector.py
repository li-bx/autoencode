

import os
import dlib
import cv2
import cv2 
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from dlib import rectangle

class FaceDetector:
    def __init__(self):
        print('cuda:', dlib.DLIB_USE_CUDA)
        dlib.DLIB_USE_CUDA=True 
        print('cuda:', dlib.DLIB_USE_CUDA)
        pwd_path = os.path.abspath(os.path.dirname(__file__))
        # 预训练的68点人脸标记点检测器
        predictor_path = os.path.join(pwd_path, "shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(predictor_path)
        # print(dlib.version)

        # print('device:', dlib.cuda.get_num_devices())
                # 使用dlib的face detector（如HOG或CNN）检测人脸位置
        self.face_detector = dlib.get_frontal_face_detector()
    def __call__(self, image):
        return self.getFace(image)

    def detect_face_landmarks(self, image):
        faces = self.face_detector(image)
        if len(faces) < 1:
            # raise ValueError("Exactly one face should be present in the image.")
            return None,None

        # 使用shape_predictor预测面部特征点
        face_shape = self.predictor(image, faces[0])
        landmarks_np = np.array([[p.x, p.y] for p in face_shape.parts()])
        rect = rectangle(landmarks_np[:,0].min(), landmarks_np[:,1].min(), landmarks_np[:,0].max(), landmarks_np[:,1].max())
        return rect, face_shape
    def get_face_mask(self,image_size, face_landmarks):
        """
        获取人脸掩模，包含轮廓
        :param image_size: 图片大小
        :param face_landmarks: 68个特征点
        :return: image_mask, 掩模图片
        """
        mask = np.zeros(image_size, dtype=np.uint8)
        points = np.concatenate([face_landmarks[0:17], face_landmarks[26:16:-1]])
        cv2.fillPoly(img=mask, pts=[points], color=1)
        return mask
    def draw_convex_hull(self,im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)
    def get_face_mask2(self,sz, landmarks):
        LEFT_EYE_POINTS = list(range(42, 48))
        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_BROW_POINTS = list(range(22, 27))
        RIGHT_BROW_POINTS = list(range(17, 22))
        NOSE_POINTS = list(range(27, 35))
        MOUTH_POINTS = list(range(48, 61))
        OVERLAY_POINTS = [
            LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
            NOSE_POINTS + MOUTH_POINTS,
        ]
        FEATHER_AMOUNT = 11
        im = np.zeros(sz, dtype=np.float64)
        #双眼的外接多边形、鼻和嘴的多边形，作为掩膜
        for group in OVERLAY_POINTS:
            self.draw_convex_hull(im,
                            landmarks[group],
                            color=1)

        # 三维掩码
        # im = np.array([im, im, im]).transpose((1, 2, 0))

        im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

        return im.astype('uint8')
    def getFace(self,image):
        """
        获取图片中的人脸
        """
        image_array = np.asarray(image)
        # 检测人脸特征点
        face_rect,landmarks = self.detect_face_landmarks(image_array)
        if face_rect is None or face_rect.height() < 10 or face_rect.width() < 10:
            return None,None,None

        # 转换为numpy数组
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        size = (image_array.shape[0], image_array.shape[1])
        mask = self.get_face_mask(size, landmarks_np)  # 脸图人脸掩模
        mask = np.stack((mask,mask,mask),axis=2)
        image_face = np.multiply(image_array,mask).astype('uint8')
        # 获取人脸区域
        image_face = image_face[face_rect.top():face_rect.bottom()+1,face_rect.left():face_rect.right()+1,:]
        # 获取人脸区域，不含额头
        return image_face,face_rect,landmarks_np

    def have_one_face(self,image):
        # 使用dlib的face detector（如HOG或CNN）检测人脸位置
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(image)
        
        print('have_one_face:',len(faces))

        if len(faces) == 1:
            return True
        else:
            return False

