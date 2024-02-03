"""
import re

# 从文件读取数据
with open('50new2.txt', 'r') as f:
    data = f.read()

# 使用正则表达式找到所有的数值
numbers = re.findall(r'tensor\(([\d.]+), device=\'cuda:0\'\)', data)

# 转为float类型
float_numbers = [float(x) for x in numbers]

# 去除重复的相邻数值
result_values = []
last_value = None
for value in float_numbers:
    if value != last_value:
        result_values.append(value)
    last_value = value

# 转为字符串并输出
result_str = ' '.join(map(str, result_values))
for item in result_values:
    print(item)
"""
from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from gensim.models import Word2Vec
import torch
import gensim
import math
dimension_ = 1024
#0825_new_embedding_w5
load_ = "10825_new_embedding_w0.pt"
save_d = "10825_new_embedding_w8.pt"
store_ = ['11024_new_embedding_w1.pt','11024_new_embedding_w2.pt','11024_new_embedding_w3.pt','11024_new_embedding_w4.pt','11024_new_embedding_w5.pt','11024_new_embedding_w6.pt','11024_new_embedding_w7.pt']
#store_ = ['10825_new_embedding_w1.pt','10825_new_embedding_w2.pt','10825_new_embedding_w3.pt','10825_new_embedding_w4.pt','10825_new_embedding_w5.pt','10825_new_embedding_w6.pt','10825_new_embedding_w7.pt','10825_new_embedding_w8.pt']
load_True = False
test_hdc = False
#MNIST_d64_onelayer
#save_ = "dimension_64_test_MNIST_binary_weight_1_.pt"
#############################################################
def piecewise_mse_loss(y_pred, y_true, threshold=45000, weight=9):
    above_threshold = y_true >= threshold
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(y_pred, y_true)
    return  torch.mean(loss + (weight-1) * above_threshold.float() * loss)

def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc

def embed(data):
    model = gensim.models.Word2Vec.load("../test_model_size/origin_embedding")
    data = [''.join(data[i:i+3]) for i in range(len(data)-3+1)]
    embeddings_ = np.zeros((len(data), 50))
    for i, token in enumerate(data):
        if token in model.wv:
            embeddings_[i] = model.wv[token]

    pos_enc = positional_encoding(len(data), 50).numpy()
    embeddings_ += pos_enc
    return embeddings_

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=10):
        ctx.constant = constant
        return torch.floor(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output), None 

Quantization_ = Quantization.apply

class AddQuantization(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        x =  torch.clamp(torch.div(torch.floor(torch.mul(tensor, 5)), 5),min=0, max=0.2)
        x = x*5
        return x

class Quantization_integer_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=1000):
        ctx.constant = constant
        x = tensor
        x_ = torch.where(x>=0 , torch.div(torch.ceil(torch.mul(x, 1)), 1), x)
        x = torch.where(x_<0 , torch.div(torch.floor(torch.mul(x_, 1)), 1), x_)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_integer = Quantization_integer_.apply

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 3, 3,3,1) # Output size will be (32, 26, 26)

        self.self_attn_1 = nn.MultiheadAttention(embed_dim=50, num_heads=5) # Self-attention layer
        self.fc1 = nn.Linear(100, 100)
        self.Bn1= nn.BatchNorm1d(100)
        self.bi_lstm1 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True, bidirectional=True)

        self.self_attn_2 = nn.MultiheadAttention(embed_dim=100, num_heads=5) # Self-attention layer
        self.fc2 = nn.Linear(200, 200)
        self.Bn2= nn.BatchNorm1d(200)
        self.bi_lstm2 = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True, bidirectional=True)


        self.fc3 = nn.Linear(200, 200) # Fully connected layer
        self.Bn3= nn.BatchNorm1d(43)
        self.bi_lstm3= nn.LSTM(input_size=200, hidden_size=200, num_layers=1, batch_first=True, bidirectional=False)


        self.fc4 = nn.Linear(200, 200) # Fully connected layer
        self.Bn4= nn.BatchNorm1d(43)
        self.bi_lstm4= nn.LSTM(input_size=200, hidden_size=200, num_layers=1, batch_first=True, bidirectional=False)

        self.fc5 = nn.Linear(200, 1) # Fully connected layer


    def forward(self, x):

        x, _ = self.self_attn_1(x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2))
        x,_ = self.bi_lstm1(x)
        x = x.permute(1, 0, 2)
        x = self.fc1(x)

        x,_ = self.self_attn_2(x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2))
        x,_ = self.bi_lstm2(x)
        x = x.permute(1, 0, 2)
        x = self.fc2(x)

        x, _ = self.bi_lstm3(x) 
        x = F.relu(self.Bn3(self.fc3(x)))
       
        x, _ = self.bi_lstm4(x) 
        x = F.relu(self.Bn4(self.fc4(x)))
        
        x = x[:, -1, :]
        x = self.fc5(x)
        
        return x


"""
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        target = target.reshape((target.shape[0],1))
        #print(target.shape)
        output = model(data)
        #print(output.shape, target.type(torch.float).shape)
        #break
        #print(output,output.shape)
        loss = F.mse_loss(output, target)
        #print(loss.shape)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
"""


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        target = target.reshape((target.shape[0],1))
        
        output = model(data)

        # 使用自定义的分段损失函数
        loss = piecewise_mse_loss(output, target)
        
        loss.backward()
        optimizer.step()

        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        #    if args.dry_run:
        #        break



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.reshape((target.shape[0],1))
            output = model(data)
            #print(output)
            correct += torch.sum(torch.abs(output-target))

    test_loss /= len(test_loader.dataset)
    correct = correct/len(test_loader.dataset)
    
    print(correct)
    return correct

def test_(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.reshape((target.shape[0],1))
            output = model(data)
            #print(output)
            for i in range(112):
                print(i,output[i],target[i])
            correct += torch.sum(torch.abs(output-target))

    test_loss /= len(test_loader.dataset)
    correct = correct/len(test_loader.dataset)
    
    print(correct)
    return correct

def test_one(model, device, data):
    model.eval()

    output = model(data)
    #print(output)
    return output



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--T', type=int, default=5, metavar='N',
                        help='SNN time window')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
                        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])

    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    
  
    
    data1 = [1., 2., 4., 1., 1., 1., 1., 1., 1., 1., 2., 3., 1., 3., 1., 1., 3.,
          1., 1., 1., 3., 4., 1., 1., 3., 1., 1., 2., 2., 2., 1., 1., 2., 4.,
          1., 1., 3., 2., 4., 1., 2., 2., 3., 1., 1.]

    data1 = [str(int(item)) for item in data1]
    #print(data1)
    data_ = ['1', '2', '4', '4', '3', '1', '4', '4', '1', '2', '2', '1', '1', '2', '2', '3', '4', '2', '4', '2', '3', '1', '3', '1', '3', '3', '3', '4', '1', '1', '4', '1', '4', '3', '2', '1', '1', '4', '1', '4', '3', '4', '1', '2', '4']
    data = embed(data1)
    data= torch.tensor(data)
    data = data.float()

    data_ = embed(data_)
    #print(data_)
    data_ = torch.tensor(data_)
    data_ = data_.float()

    data=data.reshape(1,data.shape[0],data.shape[1]).cuda()
    data_=data_.reshape(1,data_.shape[0],data_.shape[1]).cuda()
    #data=data.reshape(1,1,data.shape[0]).cuda()
    #data_=data_.reshape(1,1,data_.shape[0]).cuda()
    #print(data.shape)
    #print(test_one(model, device,data))
    #print(test_one(model, device,data_))

    
    B1 = [124]
    B2 = [111, 114, 411, 414, 342, 343, 341 ,344, 222, 223, 112, 113, 432, 433, 431, 434, 422, 423, 421, 424, 311, 314, 141, 144]
    B3 = [111, 114, 112, 113, 124, 122, 123, 121, 212, 213, 132, 133, 131, 134]
    B4 = [111, 114, 112, 113, 422, 423, 421, 424, 124, 122, 123, 121, 212, 213, 442, 443, 441, 444]
    B5 = [111, 114, 411, 414, 422, 423, 421, 424, 122, 123, 121, 212, 213, 132, 133, 131, 134, 232, 233, 231, 234, 312, 313, 142, 143]
    B6 = [111, 114, 342, 343, 341, 344, 112, 113, 422, 423, 421, 424, 311, 314, 212, 213, 132, 133, 131, 134, 412, 413, 141, 144]
    B7 = [111, 114, 411, 414, 112, 113, 422, 423, 421, 424, 122, 123, 121, 132, 133, 131, 134, 442, 443, 441, 444]
    B8 = [111, 114, 411, 414, 342, 343, 341, 344, 432, 433, 431, 434, 124, 122, 123, 121, 442, 443, 441, 444, 412, 413, 244, 141, 144]
    B9 = [222, 223, 112, 113, 432, 433, 431, 434, 422, 423, 421, 424, 311, 314, 122, 123, 121, 332, 333, 331, 334, 412, 413]
    B10 = [111, 114, 411, 414, 342, 343, 341, 344, 222, 223, 112, 113, 122, 123, 121, 132, 133, 131, 134]
    B11 = [111, 114, 411, 414, 112, 113, 311, 314]
    B12 = [411, 414, 112, 113, 422, 423, 421, 424, 212, 213, 132, 133, 131, 134, 221, 224, 322, 323, 321, 324]
    B13 = [111, 114, 112, 113, 422, 423, 421, 424, 132, 133, 131, 134, 221, 224, 322, 323, 321, 324, 412, 413]
    B14 = [111, 114, 411, 414, 112, 113, 422, 423, 421, 424, 122, 123, 121, 442, 443, 441, 444, 332, 333, 331, 334]
    B15 = [411, 414, 342, 343, 341, 344, 222, 223, 432, 433, 431, 434, 311, 314, 442, 443, 441, 444, 221, 224, 322, 323, 321, 324, 141, 144]
    # 假设data是你的二维列表数据

    c = [B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15]
    #124111111123131131113 144 311 113 112 411 324 111 311
    #124111111123131131113 144 311 113 112 411 324 111 311
    c_good = [[124],[111],[111],[123],[131],[131],[113],[144],[311],[113],[112],[411],[324],[111],[311]]
    #c_d_40 = [[124],[111],[111],[123],[131],[112],[113],[144],[311],[113],[311],[131],[324],[111],[311]]
    #c_d_46 = [[124],[111],[111],[123],[131],[113],[113],[144],[311],[113],[311],[131],[324],[111],[311]]
    c_d_62 = [[124],[111],[111],[123],[131],[113],[113],[144],[311],[113],[311],[131],[324],[111],[442]]

    #c_good = [[124],[111],[111],[123],[131],[131],[113],[411],[311],[222],[112],[411],[324],[122],[311]]
    # 遍历每一列
    la = 0
    for t in range(100000):
        l= []
        for i in range(15):  # 根据你提供的数据，共有15列
            if random.uniform(0, 1)>0.9:
                column_data = c[i]
            else:
                column_data = c_d_62[i]
            #print(column_data)
            random_choice = random.choice(column_data)
            #print(random_choice)
            digits = [int(digit) for digit in str(random_choice)]
            l.append(digits[0])
            l.append(digits[1])
            l.append(digits[2])
            #print(digits)
        #print(l)
        data_o = [str(int(item)) for item in l]
        #print(data_o)
        data1 = embed(data_o)
        tensor = torch.tensor(data1)
        tensor = tensor.to(torch.float32)
        tensor=tensor.reshape(1,tensor.shape[0],tensor.shape[1]).cuda()
        model = Net().to(device)

        model.load_state_dict(torch.load('11024_new_embedding_w0.pt'), strict=False)
        re = test_one(model, device,tensor)
        sum = re[0][0]
        #print(sum)

        for i_ in range(7):
            model.load_state_dict(torch.load(store_[i_]), strict=False)
            re = test_one(model, device,tensor)
            sum+=re[0][0]
        sum = sum/8
        if sum >la:
            la = sum
            print(la)
            print(data_o)
        #break
        #print(re)
        #print(l)
    
if __name__ == '__main__':
    main()
