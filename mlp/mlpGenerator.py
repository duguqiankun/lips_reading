import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import string

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
        image_list = sorted(image_list)
        images = []

        if len(image_list) >= self.fixed_frame_num:
            image_list = image_list[:self.fixed_frame_num]
        else:
            image_list += ["pad"] * (self.fixed_frame_num - len(image_list))

        for index,i in enumerate(image_list):
            if index%5 == 0:
                if i != "pad":
                    img = Image.open(i).convert("RGB")
                    if self.augmentation is not None:
                        img = self.augmentation(image=np.array(img, dtype=np.uint8))["image"]
                        img = Image.fromarray(img)
                else:
                    img = Image.new("RGB", (self.image_shape[1], self.image_shape[0]))

                img = self.transform(img)

                images.append(img)

        x = torch.stack(images).flatten()
        #x = torch.stack(images)


        y = torch.tensor(label_digit, dtype=torch.long)
        return x, y

if __name__ == '__main__':

    def make_char_dict():
        chars = string.ascii_lowercase
        char_dict = {"<blank>": 0}
        for idx, c in enumerate(chars):
            char_dict[c] = idx + 1
        current_len = len(list(char_dict.keys()))
        char_dict["<eos>"] = current_len
        print(char_dict)
        return char_dict


    def get_train_test_folders():
        test = open("data/eval_lst.txt", "r", encoding="utf-8").readlines()
        train = open("data/train_lst.txt", "r", encoding="utf-8").readlines()
        train_folders = [os.path.join("data", "data_aligned", i.strip("\n")) for i in train]
        test_folders = [os.path.join("data", "data_aligned", i.strip("\n")) for i in test]
        print("train videos:{}".format(len(train_folders)))
        print("test videos:{}".format(len(test_folders)))
        return train_folders, test_folders


    image_shape = (60, 60)

    char_dict = make_char_dict()
    train_folders, test_folders = get_train_test_folders()
    train_dataset = VideoDataset(
        folder_list=train_folders,
        char_dict=char_dict,
        fixed_frame_num=200,
        fixed_max_len=6,
        image_shape=image_shape,
    )
    batch_size = 5
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    for batch_idx, (imgs, label) in enumerate(train_dataloader):
        print(imgs.shape)
        print(label.shape)