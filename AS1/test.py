import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils import data
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image

class Tsdt(data.Dataset):
    def __init__(self, dt, lb):
        self.data = dt
        self.lb = lb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        label = self.lb[idx]

        return image, label

class RafDataSet(data.Dataset):
    def __init__(self, dt):
        self.data = dt
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = self.data[idx][0]
        return image

class net(nn.Module):
    def __init__(self, pretrained=False, num_classes=7, drop_rate=0):
        super(net, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        resnet.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        fc_in_dim = list(resnet.children())[-1].in_features
        self.fc = nn.Linear(fc_in_dim, num_classes)
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim,1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        att = self.alpha(x)
        out = att*self.fc(x)
        return att, out
        #return out

def sub():
    source = pd.read_csv('./AS1_data/test.csv')
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128), transforms.ToTensor()])
    processed_data = []
    print(len(source))
    print(source.iloc[10])
    for i in range(7178):
        item = source.iloc[i, 0]
        temp1 = np.array([int(j) for j in item.split(" ")])
        temp2 = np.reshape(temp1, (48,48))
        temp3 = [trans(np.uint8(temp2)) * 255, source.iloc[i, 0]]
        processed_data.append(temp3)
    batch_size = 128
    print(batch_size)
    train_loader = data.DataLoader(RafDataSet(processed_data), batch_size, shuffle=False)
    model = net(pretrained=False, drop_rate=0.2)
    cp = torch.load('./model/fin.pth')
    model.load_state_dict(cp['model_state_dict'])
    model = model.cuda()
    model.eval()
    device = "cuda:0"
    with torch.no_grad():
        for i, dt in enumerate(train_loader,0):
            inputs = dt.to(device)
            _, output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)
            if i==0 :
                arr = pred.data.cpu().numpy()
            else :
                arr = np.append(arr, pred.data.cpu().numpy())
    sb = arr.reshape(-1).astype(np.int)
    lt = np.zeros((sb.shape[0], 2))
    for i in range(sb.shape[0]):
        lt[i][0] = i+1
        lt[i][1] = sb[i]
    print(lt.shape)
    np.savetxt('ot.csv', fmt='%d', X=lt, delimiter=',')


def conf():
    source = pd.read_csv('/home/bright/Desktop/AS1/AS1/AS1_data/train.csv')
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128), transforms.ToTensor()])
    processed_data = []
    for i in range(len(source)):
        item = source.iloc[i, 1]
        temp1 = np.array([int(j) for j in item.split(" ")])
        temp2 = np.reshape(temp1, (48,48))
        temp3 = [trans(np.uint8(temp2)) * 255, source.iloc[i, 0]]
        processed_data.append(temp3)
    batch_size = 128
    print(batch_size)
    lb = np.array(source.iloc[:,0]).reshape(-1)
    train_loader = data.DataLoader(Tsdt(processed_data[-1000:], lb[-1000:]), batch_size, shuffle=False)
    model = net(pretrained=False, drop_rate=0.2)
    cp = torch.load('./md.pth')
    model.load_state_dict(cp['model_state_dict'])
    model = model.cuda()
    model.eval()
    device = "cuda:0"

    with torch.no_grad():
        for i, (dt,lb) in enumerate(train_loader,0):
            inputs = dt.to(device)
            lb = lb.to(device)
            _, output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)
            if i==0 :
                arr1 = pred.data.cpu().numpy()
                arr2 = lb.data.cpu().numpy().reshape(-1)
            else :
                arr1 = np.append(arr1, pred.data.cpu().numpy())
                arr2 = np.append(arr2, lb.data.cpu().numpy())

    pred = arr1.reshape(-1).astype(np.int)
    true = arr2.reshape(-1).astype(np.int)
    conf_mx = tf.math.confusion_matrix(labels=true, predictions=pred).numpy()
    conf_mx = np.around(conf_mx.astype('float')/conf_mx.sum(axis=1)[:,np.newaxis],decimals=2)
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    conf_mx = pd.DataFrame(conf_mx, index=emotion_dict.values(), columns=emotion_dict.values())
    figure = plt.figure(figsize=(8,8))
    sns.heatmap(conf_mx, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Prediction")
    plt.savefig("confusion_matrix.png")
    plt.show()

def sal():
    source = pd.read_csv('/home/bright/Desktop/AS1/AS1/AS1_data/train.csv')
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128), transforms.ToTensor()])
    processed_data = []
    for i in range(len(source)):
        item = source.iloc[i, 1]
        temp1 = np.array([int(j) for j in item.split(" ")])
        temp2 = np.reshape(temp1, (48,48))
        temp3 = [trans(np.uint8(temp2)) * 255, source.iloc[i, 0]]
        processed_data.append(temp3)

    model = net( drop_rate=0.2)
    cp = torch.load('./md.pth')
    model.load_state_dict(cp['model_state_dict'])
    model = model.cuda()
    model.eval()
    device = "cuda:0"

    for i in range(7):
        plot_saliency_maps(model, processed_data, i, i, 4)

def compute_saliency_map(processed_data, series, model, device="cuda:0"):
    fig = torch.unsqueeze(processed_data[series][0], dim=0)
    model.eval()
    X_var = torch.autograd.Variable(fig, requires_grad=True)
    ###########
    ###########
    ###########
    scores = model(X_var.to("cuda:0"))[1][0]
    scores.backward(torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device))
    saliency_map = X_var.grad.data
    saliency_map = saliency_map.abs()
    saliency_map, i = torch.max(saliency_map, dim=1)
    saliency_map = saliency_map.squeeze()

    return saliency_map

def plot_saliency_maps(model, processed_data, true_type, predict_type, num=5):
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad",  5: "Surprise", 6: "Neutral"}
    for i in range(0, num):
        trial = random.randint(0, len(processed_data))
        _, pred = model(processed_data[trial][0].unsqueeze(0).to("cuda:0"))
        pred = int((torch.max(pred, dim=-1)).indices.cpu())
        while (pred != predict_type or processed_data[trial][1] != true_type):
            trial = random.randint(0, len(processed_data))
            _, pred = model(processed_data[trial][0].unsqueeze(0).to("cuda:0"))
            pred = int((torch.max(pred, dim=-1)).indices.cpu())
        plt.subplot(2, num, i + 1)
        plt.imshow(processed_data[trial][0][0], cmap="gray")
        plt.axis('off')
        plt.subplot(2, num, num + i + 1)
        saliency = compute_saliency_map(processed_data, trial, model)
        plt.imshow(saliency)
        plt.axis('off')
    plt.gcf().set_size_inches(12, 5)
    plt.savefig(emotion_dict[true_type] + "(" + emotion_dict[predict_type] + ")")
    plt.show()

def calc_tar(net, layer, block, channel, pic):
    ret = pic
    for i in range(4):
        ret = net.features[i](ret)

    for j in range(4, 4+layer):
        ret = net.features[j](ret)

    if block==1 :
        ret = net.features[4+layer][0](ret)

    ret = net.features[4+layer][block].conv1(ret)

    return ret[0][channel-1]

def gradient_ascent(net, pic, lr, layer, block, channel):
    optimizer = optim.SGD([pic], lr=0.05, momentum=0.9)
    for i in range(500):
        loss = calc_tar(net, layer, block, channel, pic).sum() * (-1)
        loss.backward()
        optimizer.step()
        '''
        with torch.no_grad():
            pic.data.sub_(lr * pic.grad)
            for i in range(128):
                for j in range(128):
                    pic[0][0][i][j] %= 256

        pic.grad.zero_()
        '''
        if i%50==0 :
            print(i,'/500')

    return pic

def mv_best(net, layer, block, channel):
    X = torch.rand(size=(1,1,128,128)).to('cuda:0')*255
    X.requires_grad_(True)
    best = gradient_ascent(net,X,0.01,layer,block,channel)
    plt.subplot(3, 3, channel)
    plt.imshow(best.cpu().detach().numpy()[0][0], cmap='gray')
    plt.savefig('ga%d%d%d.png'%(layer,block,channel))
    #plt.show()


def show_ga():
    model = net( drop_rate=0.2)
    cp = torch.load('./md.pth')
    model.load_state_dict(cp['model_state_dict'])
    model = model.cuda()
    model.eval()
    for i in range(1,3):
        for j in range(0,2):
            plt.figure(33)
            for k in range(1,10):
                mv_best(model,i,j,k)
            plt.show()

'''
from lime import lime_image
from skimage.segmentation import mark_boundaries

def lm():
    source = pd.read_csv('/home/bright/Desktop/AS1/AS1/AS1_data/test.csv')
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128), transforms.ToTensor()])
    processed_data = []
    print(len(source))
    print(source.iloc[10])
    for i in range(7178):
        item = source.iloc[i, 0]
        temp1 = np.array([int(j) for j in item.split(" ")])
        temp2 = np.reshape(temp1, (48,48))
        temp3 = [trans(np.uint8(temp2)) * 255, source.iloc[i, 0]]
        processed_data.append(temp3)
    batch_size = 32
    print(batch_size)
    train_loader = data.DataLoader(RafDataSet(processed_data), batch_size, shuffle=False)
    device = "cuda:0"

    explainer = lime_image.LimeImageExplainer()

    with torch.no_grad():
        for i, dt in enumerate(train_loader,0):
            inputs = dt.to(device)
            explanation = explainer.explain_instance(np.array(dt),
                                                     batch_pred,
                                                     top_labels=3,
                                                     hide_color=0,
                                                     num_samples=32)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=3, hide_rest=False)
            break
    img_boundry = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry)
    plt.show()

def batch_pred(inputs):
    model = net(pretrained=False, drop_rate=0.2)
    cp = torch.load('./md.pth')
    model.load_state_dict(cp['model_state_dict'])
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        _, output = model(inputs)
        probs = F.softmax(output, dim=1)

    return probs.detach().cpu().numpy()
'''

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        #x = self.model.classifier(x)
        x = nn.Dropout(0.2)(x)
        x = self.model.fc(x) * self.model.alpha(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(128, 128), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            cam += w.data.cpu().detach().numpy() * target[i, :, :].data.cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam

def cam_it(layer):
    source = pd.read_csv('/home/bright/Desktop/AS1/AS1/AS1_data/train.csv')
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128), transforms.ToTensor()])
    processed_data = []
    for i in range(len(source)):
        item = source.iloc[i, 1]
        temp1 = np.array([int(j) for j in item.split(" ")])
        temp2 = np.reshape(temp1, (48,48))
        temp3 = [trans(np.uint8(temp2)) * 255, source.iloc[i, 0]]
        processed_data.append(temp3)
    batch_size = 1
    print(batch_size)
    device = 'cuda:0'
    lb = np.array(source.iloc[:,0]).reshape(-1)
    train_loader = data.DataLoader(Tsdt(processed_data[-1000:], lb[-1000:]), batch_size, shuffle=False)

    with torch.no_grad():
        for i, (dt,lb) in enumerate(train_loader,0):
            inputs = dt.to(device)
            lb = lb.to(device)
            break

    model = net(pretrained=False, drop_rate=0.2)
    cp = torch.load('./md.pth')
    model.load_state_dict(cp['model_state_dict'])
    model = model.cuda()
    model.eval()
    score_cam = ScoreCam(model, target_layer=layer)
    cam = score_cam.generate_cam(inputs, lb[0])
    plt.imshow(cam, cmap='gray')
    plt.savefig('gradcam%d.png'%layer)
    plt.show()
    if layer==0 :
        plt.imshow(inputs.cpu().detach().numpy()[0][0], cmap='gray')
        plt.savefig('gradcam_origin.png')
        plt.show()

if __name__ == '__main__' :
    sub()
    #a = net(drop_rate=0.2)
    #for module_pos, module in a.features._modules.items():
    #    print(module_pos)
    #print(a)
    #sal()
    #show_ga()
    #for i in range(8):
    #    cam_it(i)



