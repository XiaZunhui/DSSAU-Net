import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


class JointTransform2D:
    """
    for segmentation
    """
    def __init__(self, img_size=512, low_img_size=256, ori_size=512, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0,
                 p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):
        #  gamma enhancement
        # print(image.shape,mask.shape)
        image = image.transpose(1,2,0)
        mask = mask.transpose(1,2,0)
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        if image.dtype == np.uint16:
            image = image.astype(np.uint8)  # 假设图像值范围是0-65535，转换为0-255
        if mask.dtype == np.uint16:  # 如果mask也是uint16，同样需要转换
            mask = mask.astype(np.uint8)
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if self.img_size<512:
            new_h, new_w = int(self.img_size), int(self.img_size)
            image, mask = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR), F.resize(mask, (new_h, new_w),InterpolationMode.NEAREST)
        if np.random.rand() < self.p_scale:
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # transforming to tensor
        # image, mask = torch.tensor(image),torch.tensor(mask)

        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
        image = F.to_tensor(image)

        # image = image.numpy()
        # """Normalization, You can delete"""
        # image[0] = (image[0] - np.mean(image[0][np.where(image[0] > 0)])) / np.std(image[0][np.where(image[0] > 0)])
        # image[1] = (image[1] - np.mean(image[1][np.where(image[1] > 0)])) / np.std(image[1][np.where(image[1] > 0)])
        # image[2] = (image[2] - np.mean(image[2][np.where(image[2] > 0)])) / np.std(image[2][np.where(image[2] > 0)])
        # image = torch.as_tensor(image, dtype=torch.float32)

        if not self.long_mask:
            mask = F.to_tensor(mask)
            low_mask = F.to_tensor(low_mask)
        else:
            mask = to_long_tensor(mask)
            low_mask = to_long_tensor(low_mask)
        return image, mask, low_mask


class Transform2D:
    """
    for classification
    """
    def __init__(self, img_size=512, low_img_size=256, ori_size=512, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0,
                 p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image):
        image = image.transpose(1,2,0)
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image

        image = F.to_pil_image(image)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image = F.crop(image, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image = F.hflip(image)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image = F.rotate(image, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR)
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image = F.crop(image, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image = F.affine(image, *affine_params)

        image = F.to_tensor(image)

        image = image.numpy()

        """Normalization, You can delete"""
        image[0] = (image[0] - np.mean(image[0][np.where(image[0] > 0)])) / np.std(image[0][np.where(image[0] > 0)])
        image[1] = (image[1] - np.mean(image[1][np.where(image[1] > 0)])) / np.std(image[1][np.where(image[1] > 0)])
        image[2] = (image[2] - np.mean(image[2][np.where(image[2] > 0)])) / np.std(image[2][np.where(image[2] > 0)])

        image = torch.as_tensor(image, dtype=torch.float32)

        return image