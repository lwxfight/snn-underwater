import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import GaussianBlur
import numpy as np
import cv2
    
class Preprocessing2(nn.Module):
    def __init__(self, input_channels, pool_size=8, hidden_channels=64):
        super(Preprocessing2, self).__init__()
        self.input_channels = input_channels
        self.pool_size = pool_size
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)
        self.fc = nn.Sequential(
            nn.Linear(input_channels * self.pool_size * self.pool_size, hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden_channels, input_channels*input_channels + input_channels*2, bias=False),
        )
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        out_list = []
        for i in range(b):
            single_x = x[i].unsqueeze(0)  # 取出单个图像
            param = self.pool(single_x).view(1, c * self.pool_size * self.pool_size)
            param = self.act1(self.fc(param))
            single_out = torch.bmm(param[:, :c*c].view(1, 3, 3), single_x.view(1, c, -1)).view(1, c, h, w) + param[:, c * c: c*(c+1)].view(1, c, 1, 1).expand([1, c, h, w])
            single_out = torch.pow(single_out, param[:, c*(c+1):].view(1, c, 1, 1).expand([1, c, h, w]))
            out_list.append(single_out)

        out = torch.cat(out_list, dim=0)  # 合并所有增强后的图像
        out = transforms.Normalize(self.mean, self.std)(out)
        return out
    
class Defog(nn.Module):
    def __init__(self, input_channels):
        super(Defog, self).__init__()

    def zmMinFilterGray(self, src, r=7):
        '''最小值滤波，r是滤波器半径'''
        if r <= 0:
            return src
        return F.max_pool2d(-src, kernel_size=2*r+1, stride=1, padding=r).neg()

    def guidedfilter(self, I, p, r, eps):
        '''引导滤波'''
        I_mean = F.avg_pool2d(I, kernel_size=2*r+1, stride=1, padding=r)
        p_mean = F.avg_pool2d(p, kernel_size=2*r+1, stride=1, padding=r)
        Ip_mean = F.avg_pool2d(I * p, kernel_size=2*r+1, stride=1, padding=r)
        I_var = F.avg_pool2d(I * I, kernel_size=2*r+1, stride=1, padding=r) - I_mean * I_mean
        cov_Ip = Ip_mean - I_mean * p_mean

        a = cov_Ip / (I_var + eps)
        b = p_mean - a * I_mean

        a_mean = F.avg_pool2d(a, kernel_size=2*r+1, stride=1, padding=r)
        b_mean = F.avg_pool2d(b, kernel_size=2*r+1, stride=1, padding=r)

        return a_mean * I + b_mean

    def getV1(self, m, r, eps, w, maxV1):
        '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
        V1 = torch.min(m, dim=1)[0]  # 得到暗通道图像
        V1 = self.guidedfilter(V1.unsqueeze(1), self.zmMinFilterGray(V1.unsqueeze(1), 7), r, eps).squeeze(1)  # 使用引导滤波优化
        bins = 2000
        hist = torch.histc(V1, bins=bins)
        d = torch.cumsum(hist, dim=0) / V1.numel()
        lmax = (d <= 0.999).nonzero().max().item()
        A_candidates = torch.mean(m, dim=1)[V1 >= hist[lmax]]
        if A_candidates.numel() == 0:
            A = torch.mean(m, dim=[1, 2, 3]).max()
        else:
            A = A_candidates.max()
        V1 = torch.clamp(V1 * w, max=maxV1)
        return V1, A

    def forward(self, x, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
        x = x * 255
        '''去雾'''
        B, C, H, W = x.shape
        Y = torch.zeros_like(x)
        V1, A = self.getV1(x, r, eps, w, maxV1)
        for k in range(C):
            Y[:, k, :, :] = (x[:, k, :, :] - V1) / (1 - V1 / A)
        Y = torch.clamp(Y, 0, 1)
        if bGamma:
            Y = Y ** (torch.log(0.5) / torch.log(Y.mean()))
        return Y

class CLAHE(torch.nn.Module):
    def __init__(self, clip_limits, tile_grid_size):
        super(CLAHE, self).__init__()
        self.clahe_r = cv2.createCLAHE(clipLimit=clip_limits[0], tileGridSize=tile_grid_size)
        self.clahe_g = cv2.createCLAHE(clipLimit=clip_limits[1], tileGridSize=tile_grid_size)
        self.clahe_b = cv2.createCLAHE(clipLimit=clip_limits[2], tileGridSize=tile_grid_size)

    def forward(self, img):
        img_np = img.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CxHxW to HxWxC
        imgr, imgg, imgb = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        imgr, imgg, imgb = np.uint8(imgr * 255), np.uint8(imgg * 255), np.uint8(imgb * 255)

        cllr = self.clahe_r.apply(imgr)
        cllg = self.clahe_g.apply(imgg)
        cllb = self.clahe_b.apply(imgb)

        clahe_img = np.dstack((cllr, cllg, cllb))
        clahe_img = np.transpose(clahe_img, (2, 0, 1))  # HxWxC to CxHxW
        return torch.tensor(clahe_img, dtype=torch.float32)
    

class ColorBalanceAndFusionForUnderwaterImageEnhancement(nn.Module):
    def __init__(self, input_channels):
        super(ColorBalanceAndFusionForUnderwaterImageEnhancement, self).__init__()
        self.gaussian_blur = GaussianBlur(kernel_size=(7, 7), sigma=(1.0, 2.0))
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    # 论文所提出的白平衡
    def simple_color_balance(self, img, alpha, blur_need=False):
        R = img[:, 2, :, :]
        G = img[:, 1, :, :]
        B = img[:, 0, :, :]
        
        Irm = R.mean(dim=[1, 2], keepdim=True) / 256.0
        Igm = G.mean(dim=[1, 2], keepdim=True) / 256.0
        Ibm = B.mean(dim=[1, 2], keepdim=True) / 256.0
    
        Irc = R + alpha * (Igm - Irm) * (1 - Irm) * G  # 补偿红色通道
        Irc = torch.clamp(Irc, 0, 255)
        
        if blur_need:
            Ibc = B + alpha * (Igm - Ibm) * (1 - Ibm) * G  # 补偿蓝色通道
            Ibc = torch.clamp(Ibc, 0, 255)
            img = torch.stack([Ibc, G, Irc], dim=1)
        else:
            img = torch.stack([B, G, Irc], dim=1)

        img = img.type(torch.uint8)
        return img
    
    # 白平衡
    def gray_world(self, img):
        img_float = img.float()
        avg_b = img_float[:, 0, :, :].mean(dim=[1, 2], keepdim=True)
        avg_g = img_float[:, 1, :, :].mean(dim=[1, 2], keepdim=True)
        avg_r = img_float[:, 2, :, :].mean(dim=[1, 2], keepdim=True)

        gain_b = avg_g / avg_b
        gain_r = avg_g / avg_r

        balanced = torch.stack([img_float[:, 0, :, :] * gain_b, img_float[:, 1, :, :], img_float[:, 2, :, :] * gain_r], dim=1)
        balanced = torch.clamp(balanced, 0, 255).type(torch.uint8)
        return balanced
    
    # 伽马校正
    def gamma_correct(self, img, gamma=1.2):
        img_float = img.float() / 255.0
        gamma_corrected = torch.pow(img_float, gamma) * 255.0
        gamma_corrected = torch.clamp(gamma_corrected, 0, 255).type(torch.uint8)
        return gamma_corrected

    # 论文提出的锐化方法
    def sharp(self, img):
        img_float = img.float()

        im_blur = self.gaussian_blur(img_float)

        unsharp_mask = img_float - im_blur

        stretch_im = (unsharp_mask - unsharp_mask.min()) / (unsharp_mask.max() - unsharp_mask.min()) * 255.0

        result = (img_float + stretch_im) / 2
        result = torch.clamp(result, 0, 255).type(torch.uint8)
        return result
    
    # 自定义拉普拉斯算子
    def laplacian_filter(self, img):
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        img = img.unsqueeze(1)  # 将图像转换为 (B, 1, H, W) 形状
        laplacian = F.conv2d(img, kernel, padding=1)
        return laplacian.squeeze(1)  # 返回 (B, H, W) 形状的图像

    # Laplacian contrast weight (WL)
    def laplacian_weight(self, img):
        gray_img = img[:, 0, :, :] * 0.299 + img[:, 1, :, :] * 0.587 + img[:, 2, :, :] * 0.114
        laplacian = torch.abs(self.laplacian_filter(gray_img))
        return laplacian
    
    # Saliency weight (WS)
    def saliency_weight(self, img):
        gfrgb = self.gaussian_blur(img.float())
        lab = rgb_to_lab(gfrgb)
        l, a, b = lab[:, 0, :, :], lab[:, 1, :, :], lab[:, 2, :, :]
        lm, am, bm = l.mean(dim=[1, 2], keepdim=True), a.mean(dim=[1, 2], keepdim=True), b.mean(dim=[1, 2], keepdim=True)
        saliency = (l - lm) ** 2 + (a - am) ** 2 + (b - bm) ** 2
        return saliency
    
    # Saturation weight (WSat)
    def saturation_weight(self, img):
        b, g, r = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        saturation = torch.sqrt((1/3) * ((r - lum) ** 2 + (g - lum) ** 2 + (b - lum) ** 2))
        return saturation
    
    # 论文中计算权重的方法
    def weight_maps(self, img):
        WL = self.laplacian_weight(img)
        WS = self.saliency_weight(img)
        WSat = self.saturation_weight(img)
        return WL, WS, WSat

    def gaussian_pyramid(self, image, levels):
        pyramid = [image]
        for i in range(levels - 1):
            image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=True)
            pyramid.append(image)
        return pyramid

    def laplacian_pyramid(self, image, levels):
        gaussian_pyramid_list = self.gaussian_pyramid(image, levels)
        laplacian_pyramid = [gaussian_pyramid_list[-1]]

        for i in range(levels - 1, 0, -1):
            size = (gaussian_pyramid_list[i - 1].shape[2], gaussian_pyramid_list[i - 1].shape[3])
            expanded = F.interpolate(gaussian_pyramid_list[i], size=size, mode='bilinear', align_corners=True)
            laplacian = gaussian_pyramid_list[i - 1] - expanded
            laplacian_pyramid.insert(0, laplacian)
        return laplacian_pyramid

    # 金字塔重建
    def pyramid_reconstruct(self, pyramid):
        level = len(pyramid)

        for i in range(level - 1, 0, -1):
            size = (pyramid[i - 1].shape[2], pyramid[i - 1].shape[3])
            expanded = F.interpolate(pyramid[i], size=size, mode='bilinear', align_corners=True)
            pyramid[i - 1] += expanded
        output = torch.clamp(pyramid[0], 0, 255).type(torch.uint8)
        return output
    
    # 增强
    def forward(self, img, alpha=1, gamma=1.2, mode='naive', blur_need=False, level=3):
        img = img * 255
        img_balanced_color = self.simple_color_balance(img, alpha, blur_need=blur_need)
        img_balanced_white = self.gray_world(img_balanced_color)
        img_gamma_corrected = self.gamma_correct(img_balanced_white, gamma)
        img_sharpened = self.sharp(img_balanced_white)

        WL1, WS1, WSat1 = self.weight_maps(img_gamma_corrected)
        WL2, WS2, WSat2 = self.weight_maps(img_sharpened)
        W1 = (WL1 + WS1 + WSat1 + 0.1) / (WL1 + WS1 + WSat1 + WL2 + WS2 + WSat2 + 0.2)
        W2 = (WL2 + WS2 + WSat2 + 0.1) / (WL1 + WS1 + WSat1 + WL2 + WS2 + WSat2 + 0.2)

        W1 = W1.unsqueeze(1).repeat(1, 3, 1, 1)
        W2 = W2.unsqueeze(1).repeat(1, 3, 1, 1)

        if mode == 'naive':
            result = W1 * img_gamma_corrected + W2 * img_sharpened
            return result
        elif mode == 'multi':
            W1_pyramid = self.gaussian_pyramid(W1, level)
            W2_pyramid = self.gaussian_pyramid(W2, level)
            input1_laplacian_pyramid = self.laplacian_pyramid(img_gamma_corrected, level)
            input2_laplacian_pyramid = self.laplacian_pyramid(img_sharpened, level)

            result = []
            for i in range(level):
                fuse = W1_pyramid[i] * input1_laplacian_pyramid[i] + W2_pyramid[i] * input2_laplacian_pyramid[i]
                result.append(fuse)
            
            reconstructed_image = self.pyramid_reconstruct(result)
            return reconstructed_image.type(torch.uint8)
        else:
            raise ValueError('Please choose mode ("naive" or "multi")')

# Helper function to convert RGB to LAB
def rgb_to_lab(image):
    r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    mask = (r > 0.04045).float()
    r = ((r + 0.055) / 1.055) ** 2.4 * mask + r / 12.92 * (1.0 - mask)
    mask = (g > 0.04045).float()
    g = ((g + 0.055) / 1.055) ** 2.4 * mask + g / 12.92 * (1.0 - mask)
    mask = (b > 0.04045).float()
    b = ((b + 0.055) / 1.055) ** 2.4 * mask + b / 12.92 * (1.0 - mask)

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x /= 0.95047
    z /= 1.08883

    mask = (x > 0.008856).float()
    x = x ** (1 / 3) * mask + (7.787 * x + 16 / 116) * (1.0 - mask)
    mask = (y > 0.008856).float()
    y = y ** (1 / 3) * mask + (7.787 * y + 16 / 116) * (1.0 - mask)
    mask = (z > 0.008856).float()
    z = z ** (1 / 3) * mask + (7.787 * z + 16 / 116) * (1.0 - mask)

    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    lab = torch.stack([l, a, b], dim=1)
    return lab