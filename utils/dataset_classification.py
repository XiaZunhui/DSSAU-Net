import os.path

import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import cv2
from torchvision import datasets

def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def get_image_from_video(filename):
    cap = cv2.VideoCapture(filename)

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    buf = buf[:, :, :, :].transpose(0, 3, 1, 2)
    return buf


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    originSize = itkimage.GetSize()  # 获取原图size
    originSpacing = itkimage.GetSpacing()  # 获取原图spacing
    newSize = np.array(newSize, dtype='uint32')
    factor = originSize / newSize
    newSpacing = originSpacing * factor

    resampler.SetReferenceImage(itkimage)  # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    itkimgResampled.SetOrigin(itkimage.GetOrigin())
    itkimgResampled.SetSpacing(itkimage.GetSpacing())
    itkimgResampled.SetDirection(itkimage.GetDirection())
    return itkimgResampled


class DatasetClassification(Dataset):
    def __init__(self, dir, transform=None):
        self.transform = transform  # using transform in torch!
        self.pos_dir = os.path.join(dir, "pos")
        self.neg_dir = os.path.join(dir, "neg")
        images = []

        neg_samples = os.listdir(self.neg_dir)
        neg_frames = 0
        for neg_sample in neg_samples:
            frames = get_image_from_video(os.path.join(self.neg_dir, neg_sample))  # (frames,B,H,W)
            # print(frames.shape)
            for frame in frames:
                images.append(sitk.GetArrayFromImage(resize_image_itk(sitk.GetImageFromArray(frame), (256, 256, 3))))
            neg_frames += frames.shape[0]
        neg_label = np.zeros((neg_frames, 1))

        pos_samples = os.listdir(self.pos_dir)
        pos_frames = 0
        for pos_sample in pos_samples:
            frames = get_image_from_video(os.path.join(self.pos_dir, pos_sample, f"{pos_sample}.avi"))  # (frames,B,H,W)
            # print(frames.shape)
            for frame in frames:
                images.append(sitk.GetArrayFromImage(resize_image_itk(sitk.GetImageFromArray(frame), (256, 256, 3))))
            pos_frames += frames.shape[0]
        pos_label = np.ones((pos_frames, 1))

        self.images = np.array(images)
        self.labels = np.array(np.concatenate((neg_label, pos_label), axis=0)).squeeze(1)
        print(f"Image:{self.images.shape}\tLabel:{self.labels.shape}")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = correct_dims(self.images[idx])
        sample = {}
        if self.transform:
            image = self.transform(image)

        sample['image'] = image
        sample['label'] = self.labels[idx]
        return sample

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DatasetClassification2(Dataset):
    def __init__(self, dir, transform=None, image_size=(256, 256)):
        self.transform = transform  # 可选的图像转换
        self.image_size = image_size  # 目标图像尺寸
        self.image_paths = []
        self.labels = []

        # 处理pos文件夹中的图像
        pos_dir = os.path.join(dir, "pos")
        pos_images = [os.path.join(pos_dir, img) for img in os.listdir(pos_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_paths.extend(pos_images)
        self.labels.extend([1] * len(pos_images))

        # 处理neg文件夹中的图像
        neg_dir = os.path.join(dir, "neg")
        neg_images = [os.path.join(neg_dir, img) for img in os.listdir(neg_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_paths.extend(neg_images)
        self.labels.extend([0] * len(neg_images))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # 加载图像并转换为RGB

        # 调整图像尺寸
        if self.image_size:
            image = image.resize(self.image_size)

        # 如果未传入transform, 自动将image转成tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  # 默认转换为张量

        label = self.labels[idx]
        return image, label

def MyDataset(root):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小为 256x256
        transforms.ToTensor()  # 将图像转换为张量
    ])
    dataset = datasets.ImageFolder(root, transform)
    return dataset

class DatasetClassificationLow(Dataset):
    def __init__(self, dir, transform=None):
        self.transform = transform  # using transform in torch!
        self.pos_dir = os.path.join(dir, "pos")
        self.neg_dir = os.path.join(dir, "neg_low")
        images = []

        neg_samples = os.listdir(self.neg_dir)
        neg_frames = 0
        for neg_sample in neg_samples:
            frames = get_image_from_video(os.path.join(self.neg_dir, neg_sample))  # (frames,B,H,W)
            # print(frames.shape)
            for frame in frames:
                images.append(frame)
            neg_frames += frames.shape[0]
        neg_label = np.zeros((neg_frames, 1))

        pos_samples = os.listdir(self.pos_dir)
        pos_frames = 0
        for pos_sample in pos_samples:
            frames = get_image_from_video(
                os.path.join(self.pos_dir, pos_sample, "low", f"{pos_sample}.avi"))  # (frames,B,H,W)
            # print(frames.shape)
            for frame in frames:
                images.append(frame)
            pos_frames += frames.shape[0]
        pos_label = np.ones((pos_frames, 1))

        self.images = np.array(images)
        self.labels = np.array(np.concatenate((neg_label, pos_label), axis=0)).squeeze(1)
        print(f"Image:{self.images.shape}\tLabel:{self.labels.shape}")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = correct_dims(self.images[idx])
        sample = {}
        if self.transform:
            image = self.transform(image)

        sample['image'] = image
        sample['label'] = self.labels[idx]
        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from augmentation import Transform2D

    tf = Transform2D(p_flip=1, crop=None)
    dataset = DatasetClassification("../val", tf)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for idx, sample in enumerate(dataloader):
        if sample['label'].sum():
            image = sample['image']
            label = sample['label']
            print(image.shape, label.shape)
            plt.imshow(image[0][0], cmap="gray")
            plt.show()
            break
