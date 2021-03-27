import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

path = '../input/njunnfin1'

def load_data(data, label=True):
    if label:
        return list(data['comment']), [y-1 for y in data['value']]
    return list(data['comment'])

class CommentDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)

def convert_text_to_ids(tokenizer, text, max_len=20):
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text, max_length=max_len, truncation=True, add_special_tokens=True)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
    else:
        input_ids = []
        token_type_ids = []
        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, truncation=True, add_special_tokens=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    return input_ids, token_type_ids


def seq_padding(tokenizer, X):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if len(X) <= 1:
        return torch.tensor(X)

    L = [len(x) for x in X]
    ML = max(L)
    X = torch.Tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X])
    return X

def bert_training(batch_size, n_epoch, lr, train, valid, model, tokenizer, model_dir, device):
    train_loss = []
    valid_loss = []
    train_accs = []
    valid_accs = []

    model.train()
    t_batch = len(train)
    v_batch = len(valid)
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(train):
            input_ids, token_type_ids = convert_text_to_ids(tokenizer, inputs)
            input_ids = seq_padding(tokenizer, input_ids)
            token_type_ids = seq_padding(tokenizer, token_type_ids)
            labels = labels.squeeze()
            input_ids, token_type_ids, labels = input_ids.long(), token_type_ids.long(), labels.long()
            optimizer.zero_grad()
            input_ids, token_type_ids, labels = input_ids.to(device), token_type_ids.to(device), labels.to(device)

            output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=labels)
            y_pred_prob = output[1]
            y_pred_labels = y_pred_prob.argmax(dim=1)

            loss = output[0]
            loss.backward()
            optimizer.step()

            correct =((y_pred_labels == labels.view(-1)).sum()).item()
            total_acc += (correct / batch_size)
            total_loss += loss.item()
        print('[ Epoch{} ]: \nTrain | Loss:{:.5f} Acc: {:.3f}'.format(epoch+1, total_loss/t_batch, total_acc/t_batch*100))
        train_loss.append(total_loss/t_batch)
        train_accs.append(total_acc/t_batch)

        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                input_ids, token_type_ids = convert_text_to_ids(tokenizer, inputs)
                input_ids = seq_padding(tokenizer, input_ids)
                token_type_ids = seq_padding(tokenizer, token_type_ids)
                labels = labels.squeeze()
                input_ids, token_type_ids, labels = input_ids.long(), token_type_ids.long(), labels.long()
                input_ids, token_type_ids, labels = input_ids.to(device), token_type_ids.to(device), labels.to(device)

                output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=labels)
                y_pred_prob = output[1]
                y_pred_labels = y_pred_prob.argmax(dim=1).squeeze()

                loss = output[0]
                correct = ((y_pred_labels == labels.view(-1)).sum()).item()
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            valid_loss.append(total_loss/v_batch)
            valid_accs.append(total_acc/v_batch)

            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, "{}".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))

        print('-----------------------------------------------')
        model.train()

    return train_loss, train_accs, valid_loss, valid_accs

batch_size =128
epoch = 10
lr = 1e-3
device = torch.device("cuda")

train_data = pd.read_csv(os.path.join(path,'train.csv'),index_col=0)
train_unlabeled_data = pd.read_csv(os.path.join(path,'train_unlabeled.csv'),index_col=0)
test_data = pd.read_csv(os.path.join(path, 'test.csv'),index_col=0)

train_x, train_y = load_data(train_data)
train_unlabeled_x = load_data(train_unlabeled_data, False)
test_x = load_data(test_data, False)

model_dir = os.path.join(path, 'bert0.param')

X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], train_y[:180000], train_y[180000:]

train_dataset = CommentDataset(X=X_train, y=y_train)
val_dataset = CommentDataset(X=X_val, y=y_val)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

config = BertConfig.from_pretrained("bert-base-chinese", num_labels=2)
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", config=config)
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_loss, train_accs, valid_loss, valid_accs = bert_training(batch_size, epoch, lr, train_loader, val_loader, model, tokenizer, model_dir, device)


fig = plt.figure()
plt.plot(list(range(1, 1+epoch)), train_accs, label='train_acc')
plt.plot(list(range(1, 1+epoch)), valid_accs, label='valid_acc')
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(list(range(1, 1+epoch)), train_loss, label='train_loss')
plt.plot(list(range(1, 1+epoch)), valid_loss, label='valid_loss')
plt.legend()
plt.show()

