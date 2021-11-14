from torch.autograd.grad_mode import no_grad
import data_loader
import torch.nn as nn
import mlp_crnn
import torch
import torch.utils.data as Data

import random

def main():
    bs = 5

    ttl_epochs = 200



    net = mlp_crnn.videoModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': net.parameters()}],lr=0.1,momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        verbose=True,
                                                        factor=0.5,
                                                        patience=5,
                                                        threshold=0.00001)

    net.cuda()


    for epoch in range(ttl_epochs):
        DATASET = data_loader.TrainData('./',skip=random.randrange(5))
        DATALOADER = Data.DataLoader(DATASET,batch_size=bs,shuffle=True,num_workers=4)

        running_loss = 0
        net.train()

        for batch_idx, (imgs, labels) in enumerate(DATALOADER):
            imgs = imgs.cuda()
            labels = labels.cuda()

            out = net(imgs)

            out = out.view(out.shape[0]*out.shape[1],-1)
            
            labels = labels.view(labels.shape[0]*labels.shape[1])
            
            loss = criterion(out,labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.detach().item()


            if batch_idx%10 ==0:
                print(f'epoch: {epoch}  batches: {batch_idx}   training loss:{running_loss/(batch_idx+1)} ')

        #eval

        TestData = data_loader.TestData('./',skip=random.randrange(5))
        Test_DATALOADER = Data.DataLoader(TestData,batch_size=bs,shuffle=True,num_workers=4)


        net.eval()
        test_running_loss = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(Test_DATALOADER):
                imgs = imgs.cuda()
                labels = labels.cuda()

                out = net(imgs)

                out = out.view(out.shape[0]*out.shape[1],-1)
                
                labels = labels.view(labels.shape[0]*labels.shape[1])
                
                loss = criterion(out,labels)

                test_running_loss += loss.detach().item()

            print('================================val===============================================')
            print(f'val: {epoch}  batches: {batch_idx}   val loss:{test_running_loss/(batch_idx+1)} ')
            print('================================val===============================================')

        scheduler.step(test_running_loss/(batch_idx+1))

if __name__ == '__main__':
    main()