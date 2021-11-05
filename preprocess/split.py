import os
from random import shuffle


my_list = os.listdir('data/data_aligned')

shuffle(my_list)

train_list = my_list[:-20]

eval_list = my_list[-20:]

print(eval_list)


train_file = open('train_lst.txt','w')

for clip in train_list:
    train_file.write(clip+'\n')

eval_file = open('eval_lst.txt','w')

for clip in eval_list:
    eval_file.write(clip+'\n')