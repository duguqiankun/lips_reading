import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
import torchvision.transforms as transforms


augmentation = A.Compose([
    A.OneOf([
        A.IAAAdditiveGaussianNoise(p=0.9),
        A.GaussNoise(p=0.9),
    ], p=0.9),
    A.OneOf([
        A.MotionBlur(p=0.9),
        A.MedianBlur(blur_limit=3, p=0.9),
        A.Blur(blur_limit=4, p=0.9),
    ], p=0.9),
    A.OneOf([
        A.CLAHE(clip_limit=2, p=0.9),
        A.IAASharpen(p=0.9),
        A.IAAEmboss(p=0.9),
        A.RandomBrightnessContrast(p=0.95),
    ], p=0.9),
    A.OneOf(
        [
            A.HueSaturationValue(p=0.9),
            A.RandomGamma(p=0.9),
            A.IAAPerspective(p=0.05),
        ], p=0.9,
    )
])


def build_transform(shape):
    transform = transforms.Compose([
        transforms.Resize((shape[0], shape[1])),
        transforms.ToTensor()
    ])
    return transform


class VideoDataset(Dataset):
    def __init__(self, folder_list, char_dict,
                 fixed_frame_num=200, fixed_max_len=6,
                 image_shape=(100, 100),
                 aug=augmentation):
        self.folders = folder_list
        np.random.shuffle(self.folders)
        self.fixed_frame_num = fixed_frame_num
        self.char_dict = char_dict
        self.fixed_max_len = fixed_max_len
        self.augmentation = aug
        self.image_shape = image_shape
        self.transform = build_transform(shape=self.image_shape)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        image_folder = self.folders[index]
        label = image_folder.split("/")[-1].split("_")[-1].strip(" ")
        label_digit = [self.char_dict[i] for i in label]
        assert len(label_digit) < self.fixed_max_len
        label_digit.append(self.char_dict["<eos>"])
        rest = self.fixed_max_len - len(label_digit)
        if rest:
            label_digit += [self.char_dict["<blank>"]] * rest

        image_list = [os.path.join(image_folder, i) for i in os.listdir(image_folder) if i.endswith(".jpg")]
        image_list = sorted(image_list, reverse=False)
        images = []

        if len(image_list) >= self.fixed_frame_num:
            image_list = image_list[:self.fixed_frame_num]
        else:
            image_list += ["pad"] * (self.fixed_frame_num - len(image_list))

        for i in image_list:
            if i != "pad":
                img = Image.open(i).convert("RGB")
                if self.augmentation is not None:
                    img = self.augmentation(image=np.array(img, dtype=np.uint8))["image"]
                    img = Image.fromarray(img)
            else:
                img = Image.new("RGB", (self.image_shape[1], self.image_shape[0]))

            img = self.transform(img)
            images.append(img)
        x = torch.stack(images)
        y = torch.tensor(label_digit, dtype=torch.long)
        return x, y


