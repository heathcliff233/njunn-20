import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn


class RafDataSet(data.Dataset):
    def __init__(self, dt, lb):
        self.data = dt
        self.label = lb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        label = self.label[idx]

        return image, label, idx

class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 7, drop_rate = 0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate

        resnet  = models.resnet18(pretrained)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1
        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512

        #vgg19_bn
        '''
        resnet = models.vgg19_bn(pretrained)
        resnet.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features = nn.Sequential(*list(resnet.children())[:-1], *list(resnet.classifier[:-1]))
        fc_in_dim = resnet.classifier[-1].in_features
        '''


        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())



    def forward(self, x):
        x = self.features(x)

        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out


def run_training():

    imagenet_pretrained = False
    res18 = Res18Feature(pretrained = imagenet_pretrained, drop_rate = 0.2)

    source = pd.read_csv("AS1_data/train.csv")
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128), transforms.ToTensor()])
    processed_data = []
    for i in range(0, len(source)):
        item = source.iloc[i, 1]
        temp1 = np.array([int(j) for j in item.split(" ")])
        temp2 = np.reshape(temp1, (48,48))

        temp3 = [trans(np.uint8(temp2)) * 255, source.iloc[i, 0]]
        processed_data.append(temp3)

    lb = np.array(source.iloc[:,0]).reshape(-1)
    lr, num_epochs, batch_size = 0.001, 100, 64
    train_loader = data.DataLoader(RafDataSet(processed_data[:int(len(processed_data) * 0.9)], lb[:int(len(processed_data)*0.9)]), batch_size, shuffle=True)
    val_loader = data.DataLoader(RafDataSet(processed_data[int(len(processed_data) * 0.9):], lb[int(len(processed_data) * 0.9):]), batch_size, shuffle=False)

    params = res18.parameters()


    optimizer = torch.optim.SGD(params, 0.05,
                                    momentum=0.8,
                                    weight_decay = 1e-4)


    #base_optimizer = torch.optim.SGD
    #optimizer = SAM(params, base_optimizer, lr=0.1, momentum=0.9, weight_decay = 1e-4)


    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    res18 = res18.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    margin_1 = 0.07
    margin_2 = 0.2
    beta = 0.7

    print(next(iter(train_loader)))

    os.system("mkdir models")
    for i in range(1, 20 + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        res18.train()
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            batch_sz = imgs.size(0)
            iter_cnt += 1
            tops = int(batch_sz* beta)
            optimizer.zero_grad()
            imgs = imgs.cuda()
            targets = targets.cuda()

            #add for SAM

            #_, ot = res18(imgs)
            #criterion(ot, targets).backward()
            #optimizer.first_step(zero_grad=True)


            attention_weights, outputs = res18(imgs)

            # Rank Regularization
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest = False)

            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            diff  = low_mean - high_mean + margin_1

            if diff > 0:
                RR_loss = diff
            else:
                RR_loss = 0.0


            loss = criterion(outputs, targets) + RR_loss
            loss.backward()
            #add for SAM
            optimizer.step()
            #optimizer.second_step(zero_grad=True)

            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

            # Relabel samples
            if i >= 12:
                sm = torch.softmax(outputs, dim = 1)
                Pmax, predicted_labels = torch.max(sm, 1) # predictions
                Pgt = torch.gather(sm, 1, targets.view(-1,1)).squeeze() # retrieve predicted probabilities of targets
                true_or_false = Pmax - Pgt > margin_2
                update_idx = true_or_false.nonzero().squeeze() # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
                label_idx = indexes[update_idx] # get samples' index in train_loader
                relabels = predicted_labels[update_idx] # predictions where (Pmax - Pgt > margin_2)
                train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader

        scheduler.step()
        acc = correct_sum.float() / float(len(processed_data)*0.9)
        running_loss = running_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            res18.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                _, outputs = res18(imgs.cuda())
                targets = targets.cuda()
                loss = criterion(outputs, targets)
                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(outputs, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)

            running_loss = running_loss/iter_cnt
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))

            if acc > 0.6 :
                torch.save({'iter': i,
                            'model_state_dict': res18.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('models', "epoch"+str(i)+"_acc"+str(acc)+".pth"))
                print('Model saved.')


if __name__ == '__main__':
    run_training()
