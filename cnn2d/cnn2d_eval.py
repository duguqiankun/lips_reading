import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from cnn2d_image_generator import VideoDataset
# from image_2dcrnn import VideoModel
import torch
from torch.utils.data import DataLoader
import string
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

image_shape = (60, 60)

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

def compute_val_acc(scores, y):
    num = scores.size(0)
    prediction = scores.argmax(dim=1)
    indicator = (prediction == y)
    num_matches = indicator.sum()
    return num_matches.float() / num

char_dict = make_char_dict()
train_folders, test_folders = get_train_test_folders()
train_dataset = VideoDataset(
    folder_list=train_folders,
    char_dict=char_dict,
    fixed_frame_num=200,
    fixed_max_len=6,
    image_shape=image_shape,
)
batch_size = 10
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = VideoDataset(
    folder_list=test_folders,
    char_dict=char_dict,
    fixed_frame_num=200,
    fixed_max_len=6,
    aug=None,  # No need to do data augmentation in testing dataset
    image_shape=image_shape,
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

model = torch.load("2dcrnn_model.pkl")

model.eval()
acc = 0
count = 0
with torch.no_grad():
    for idx, data in enumerate(test_dataloader):
        x, y = data
        size = y.size()
        x = x.to(device)
        y = y.to(device)
        scores = model(x)

        scores = scores.view(size[0] * size[1], -1)
        y = y.view(size[0] * size[1])
        acc += compute_val_acc(scores, y)
        count += 1

print("Acc in inference process is {}".format(acc / count))