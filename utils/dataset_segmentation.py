import os.path
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import cv2


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


def get_image_from_video(filename, frame):
    cap = cv2.VideoCapture(filename)
    print(f'正在处理：{filename}')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((len(frame), frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    index = 0
    while (fc < frameCount and ret):
        if fc in frame:
            # print(fc)
            ret, buf[index] = cap.read()
            index += 1
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


class DatasetSegmentation(Dataset):
    """
    We didn't use semi-supervised learning, this Dataset Class just load labelled datas.
    If you want to try semi-supervised learning to make full use of unlabelled datas, please design your own Dataset Class!
    """

    def __init__(self, dir, transform=None):
        self.transform = transform  # using transform in torch!
        self.dir = dir
        self.files = os.listdir(dir)
        labels = []
        images = []
        for file in self.files:
            path = os.path.join(dir, file, "mask")
            frame = [int(i.split("_")[1]) for i in os.listdir(path)]
            frame = sorted(frame)
            # print(frame)
            image = get_image_from_video(os.path.join(dir, file, f"{file}.avi"), frame)
            images.append(np.array(image))
            # print(sorted(os.listdir(path), key=lambda x: int(x.split("_")[1])))
            for pic in sorted(os.listdir(path), key=lambda x: int(x.split("_")[1])):
                label = sitk.ReadImage(os.path.join(dir, file, "mask", pic))
                label = sitk.GetArrayFromImage(label)
                label[np.where(label == 7)] = 1
                label[np.where(label == 8)] = 2
                labels.append(label)

        images = np.concatenate(images, axis=0)
        self.images = images
        self.labels = np.array(labels)
        print(f"Image:{self.images.shape}\tLabel:{self.labels.shape}")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = correct_dims(self.images[idx])
        label = np.array([self.labels[idx]])
        sample = {}
        if self.transform:
            image, label, low_label = self.transform(image, label)

        sample['image'] = image
        sample['label'] = label
        sample['low_label'] = low_label
        return sample


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from augmentation import JointTransform2D

    tf = JointTransform2D(p_flip=1, crop=None, long_mask=True)
    dataset = DatasetSegmentation("../dataset_sample/pos", tf)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for idx, sample in enumerate(dataloader):
        if idx == 1:
            image = sample['image']
            label = sample['label'].squeeze(1)
            print(image.shape, label.shape)
            # plt.imshow(image[0][0],cmap="gray")
            plt.imshow(label[0], cmap="gray")
            plt.show()
