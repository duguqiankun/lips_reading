import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mlp_Generator import VideoDataset
from mlp_model import VideoModel
import torch
from torch.utils.data import DataLoader
import string
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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

batch_size = 1

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
model = VideoModel(number_classes=len(list(char_dict.keys())),max_len=6,image_shape=image_shape)

model = model.to(device)

model.load_state_dict(torch.load('mlp/weights/best.pt'))

model.eval()
with torch.no_grad():
    for idx, data in enumerate(test_dataloader):
        x, y = data

        size = y.size()
        x = x.to(device)
        y = y.to(device)
        scores = model(x)
        print('========================')

        print('predict',torch.argmax(scores,dim=2))
        
        print('goundtruch',y)

        print('========================')