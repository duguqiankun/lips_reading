import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mlpGenerator import VideoDataset
from mlpvideo_crnn import VideoModel
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
model = VideoModel(number_classes=len(list(char_dict.keys())),
                   max_len=6,
                   image_shape=image_shape)
model = model.to(device)
print(model)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9)

steps_per_epoch = len(train_folders) // 10 + 1
epochs = 300
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       verbose=True,
                                                       factor=0.1,
                                                       patience=5,
                                                       threshold=0.00001)


def train_process():
    running_loss = 0
    num_batches = 0

    model.train()
    for idx, data in enumerate(train_dataloader):
        optimizer.zero_grad()

        x, y = data
        size = y.size()
        x = x.to(device)
        y = y.to(device)

        x.requires_grad_()

        scores = model(x)

        scores = scores.view(size[0] * size[1], -1)
        y = y.view(size[0] * size[1])
        loss = criterion(scores, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        running_loss += loss.detach().item()
        num_batches += 1
        print("time:{}, epoch: {} step: {}, avg running loss is {}".format(
            time.ctime(), epoch + 1, idx + 1, running_loss / num_batches
        ))
    return running_loss, num_batches


def testing_process():
    running_loss = 0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            x, y = data
            size = y.size()
            x = x.to(device)
            y = y.to(device)
            scores = model(x)

            scores = scores.view(size[0] * size[1], -1)
            y = y.view(size[0] * size[1])
            loss = criterion(scores, y)
            running_loss += loss.item()
            num_batches += 1
    return running_loss, num_batches


eval_loss = 3

for epoch in range(epochs):
    running_loss, num_batches = train_process()
    test_running_loss, test_num_batches = testing_process()
    print("*" * 100)
    print("epoch: {}, avg training loss:{}, avg validation loss:{}".format(epoch + 1, running_loss / num_batches,
                                                                           test_running_loss / test_num_batches))
    scheduler.step(test_running_loss / test_num_batches)
    print("*" * 100)

    if test_running_loss / test_num_batches < eval_loss:
        torch.save(model.state_dict(),'mlp/weights/best.pt')
        eval_loss =  test_running_loss / test_num_batches