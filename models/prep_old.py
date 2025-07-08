# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : Zhejiang University

from tkinter import NO
import numpy as np
import cv2

#封装成类
class Color_Balance_and_Fusion_for_Underwater_Image_Enhancement():
    def __init__(self):
        pass

    #读图片
    def read_img_by_cv2(self,img_path):
        img=cv2.imread(img_path)
        # img=cv2.resize(img,(800,600))
        return img
    
    #论文所提出的白平衡
    def simple_color_balance(self,img,alpha,blur_need=None):
        R = img[:, :, 2]
        G = img[:, :, 1]
        B = img[:, :, 0]
        # 三颜色通道均值再归一化，对应 I¯r I¯g I¯b
        Irm = np.mean(R)/256.0
        Igm = np.mean(G)/256.0
        Ibm = np.mean(B)/256.0
    
        Irc = R + alpha * (Igm-Irm)*(1-Irm)*G  # 补偿红色通道
        Irc = np.array(Irc.reshape(G.shape), np.uint8)
        if blur_need:
            Ibc = B + alpha * (Igm-Ibm)*(1-Ibm)*G  # 补偿蓝色通道
            Ibc = np.array(Ibc.reshape(G.shape), np.uint8)
            img = cv2.merge([Ibc, G,Irc])
        else:
            img = cv2.merge([B,G,Irc])

        img=np.clip(img,0,255)
        img=img.astype(np.uint8)
        return img
    
    #白平衡
    def gray_world(self,img):
    # 将图像转换为浮点格式
        img_float = img.astype(float)
        # 计算图像的各通道平均值
        avg_b = np.mean(img_float[:, :, 0])
        avg_g = np.mean(img_float[:, :, 1])
        avg_r = np.mean(img_float[:, :, 2])

        # 计算各通道的增益
        gain_b = avg_g / avg_b
        gain_r = avg_g / avg_r

        # 应用增益来进行白平衡
        balanced = cv2.merge([img_float[:, :, 0] * gain_b, img_float[:, :, 1], img_float[:, :, 2] * gain_r])

        # 将结果限制在0到255的范围内
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        return balanced
    
    #伽马校正
    def gamma_correct(self,img,gamma=1.2):
        # 进行伽马校正
        gamma_corrected = np.power(img / 255.0, gamma) * 255.0
        gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
        return gamma_corrected

    #论文提出的锐化方法
    def sharp(self,image):
        # Convert the image to floating point format
        image = image.astype(np.float32)

        # Apply Gaussian smoothing
        im_blur = cv2.GaussianBlur(image,(7,7),0)

        # Calculate unsharp mask
        unsharp_mask = image - im_blur

        # Perform histogram stretching
        stretch_im = cv2.normalize(unsharp_mask, None, 0, 255, cv2.NORM_MINMAX) # type: ignore

        # Combine original and stretched images
        result = (image + stretch_im)/2

        # Convert result back to uint8 format
        result = np.clip(result, 0, 255).astype(np.uint8)
    
        return result
    
    # Laplacian contrast weight (WL)
    def laplacian_weight(self,img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        w = cv2.Laplacian(img, cv2.CV_64F)
        w = cv2.convertScaleAbs(w)
        return w
    
    # Saliency weight (WS)
    def saliency_weight(self,img):
        gfrgb = cv2.GaussianBlur(img,(3,3),0)
        lab = cv2.cvtColor(gfrgb,cv2.COLOR_RGB2LAB)
        l = np.double(lab[:, :, 0])
        a = np.double(lab[:, :, 1])
        b = np.double(lab[:, :, 2])
        lm = np.mean(np.mean(l))
        am = np.mean(np.mean(a))
        bm = np.mean(np.mean(b))
        w = np.square(l-lm) + np.square(a-am) + np.square((b-bm))
        return w
    
    # Saturation weight (WSat)
    def saturation_weight(self,img):
        b,g,r=cv2.split(img)
        lum=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        w=np.sqrt((1/3)*((r-lum)**2+(g-lum)**2+(b-lum)**2))
        return w
    
    # 论文中计算权重的方法
    def weight_maps(self,img):
        img.astype(np.float64)
        WL = self.laplacian_weight(img)
        WS = self.saliency_weight(img)
        WSat = self.saturation_weight(img)
        return WL, WS, WSat

    def gaussian_pyramid(self,image, levels):
        pyramid = [image]
        for i in range(levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
            
        return pyramid

    def laplacian_pyramid(self,image, levels):
        gaussian_pyramid_list = self.gaussian_pyramid(image, levels)
        laplacian_pyramid = [gaussian_pyramid_list[-1]]

        for i in range(levels - 1, 0, -1):
            size = (gaussian_pyramid_list[i - 1].shape[1], gaussian_pyramid_list[i - 1].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid_list[i], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid_list[i - 1], expanded)
            laplacian_pyramid.insert(0, laplacian)
        return laplacian_pyramid

    # 金字塔重建
    def pyramid_reconstruct(self,pyramid):
        level = len(pyramid)

        for i in range(level - 1, 0, -1):
            m, n, c  = pyramid[i - 1].shape
            pyramid[i - 1] += pyramid[i - 1] + cv2.resize(pyramid[i], (n, m))
        output = pyramid[0]
        output=np.clip(output, 0, 255).astype(np.uint8)
        return output
    
    # 增强
    def Enhance(self,img,alpha,gamma,mode='naive',blur_need=False,level=3):

        img_balenced_color=self.simple_color_balance(img,alpha,blur_need=blur_need)

        img_balenced_white=self.gray_world(img_balenced_color)

        img_gamma_corrected=self.gamma_correct(img_balenced_white,gamma)
        img_sharpen=self.sharp(img_balenced_white)

        (WL1, WS1, WSat1) = self.weight_maps(img_gamma_corrected)
        (WL2, WS2, WSat2) = self.weight_maps(img_sharpen)
        W1 = (WL1 + WS1 + WSat1+0.1)/(WL1 + WS1 + WSat1 + WL2 + WS2 + WSat2+0.2)
        W2 = (WL2 + WS2 + WSat2+0.1)/(WL1 + WS1 + WSat1 + WL2 + WS2 + WSat2+0.2)

        W1=np.repeat(W1[:, :, np.newaxis], 3, axis=-1)
        W2=np.repeat(W2[:, :, np.newaxis], 3, axis=-1)

        if mode=='naive':
            result=np.multiply(W1,img_gamma_corrected)+ np.multiply(W2,img_sharpen)
            return result.astype(np.uint8)
        elif mode=='multi':
            # 构造权值的高斯金字塔
           
            W1 = self.gaussian_pyramid(W1,level)
            W2 = self.gaussian_pyramid(W2,level)
            
            # 构造输入图像的拉普拉斯金字塔
            input1_laplacian_pyramid=self.laplacian_pyramid(img_gamma_corrected,level)
            input2_laplacian_pyramid=self.laplacian_pyramid(img_sharpen,level)

            # 权值融合
            result = []
            for i in range(level):
                fuse=np.multiply(W1[i],input1_laplacian_pyramid[i])+ np.multiply(W2[i],input2_laplacian_pyramid[i])
                result.append(fuse)
            
            # 图像重建
            reconstructed_image = self.pyramid_reconstruct(result)
            return reconstructed_image.astype(np.uint8)
        
        else:
            print('Please choose mode ("naive" or "multi")')


#Example
def prepexample():
    enhancer=Color_Balance_and_Fusion_for_Underwater_Image_Enhancement()
    img_path=r'E:\underwater\Color Balance and Fusion for Underwater Image Enhancement\matlib\image\5.jpg'
    img=cv2.imread(img_path)
    img=cv2.resize(img,(600,450))
    img_enhanced_naive=enhancer.Enhance(img,mode='naive',alpha=1,gamma=2,blur_need=False)
    img_enhanced_multi=enhancer.Enhance(img,mode='multi',alpha=1,gamma=2,blur_need=False,level=3)

    cv2.imshow('result',np.hstack((img,img_enhanced_naive,img_enhanced_multi))) # type: ignore
    cv2.waitKey()
    cv2.destroyAllWindows()

