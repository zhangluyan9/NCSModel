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
#4 rounds: 10825_new_embedding_w8 (1-7)
# 5 rounds: 10903_new_embedding_w8 (1-7)
load_ = "../7round/11018_new_embedding_w0.pt"
save_d = "11024_new_embedding_w7111.pt"
load_True = True
#MNIST_d64_onelayer
#save_ = "dimension_64_test_MNIST_binary_weight_1_.pt"
#############################################################
def piecewise_mse_loss(y_pred, y_true, threshold=65000, scale=8):
    
    above_threshold = y_true >= threshold
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(y_pred, y_true)
    return  torch.mean(loss + (scale-1) * above_threshold.float() * loss)

def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc

def embed(data):
    model = gensim.models.Word2Vec.load("origin_embedding")
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
        #print(x.shape)
        #print(x.shape)
        #x= x.mean(dim=1)
        #print(x.shape)

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

        loss = piecewise_mse_loss(output, target)
        
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    co = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #target = target.reshape((target.shape[0],1))
            output = model(data)
            #print(output.shape)
            output = output.reshape((output.shape[0]))
            #print(output.shape)

            correct += torch.sum(torch.abs(output-target))
            condition = (output > 45000)
            co+=torch.sum(condition)
            selected_output = output[condition]
            selected_target = target[condition]

            #correct += (torch.sum(torch.abs(selected_output - selected_target)))/torch.sum(condition)
            
    test_loss /= len(test_loader.dataset)
    correct = correct/len(test_loader.dataset)
    
    print(correct,co)
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
    
    path = './new_encoding_training_1025.npz'  
    f = np.load(path)
    test_x, test_y = f['train_x'], f['label_x']
    X_test_ = torch.FloatTensor(test_x)
    #print(X_train_.shape)
    #X_train_ = X_train_.reshape(-1,1,45)
    y_test_ = torch.FloatTensor(test_y)

    #new_encoding_training
    # 4 rounds: new_encoding_training_0726_1
    # 5 rounds: 
    path = './new_encoding_training_1025.npz'  
    f = np.load(path)
    train_x, train_y = f['train_x'], f['label_x']
    X_train_ = torch.FloatTensor(train_x)
    #print(X_train_.shape)
    #X_train_ = X_train_.reshape(-1,1,45)
    y_train_ = torch.FloatTensor(train_y)
    #y_train_ = y_train_.reshape(-1,1,180,1)

    #print(X_train_[0],y_train_[0])
    #print(X_train_[72999],y_train_[72999])

    torch_dataset_train = torch.utils.data.TensorDataset(X_train_, y_train_)
    torch_dataset_test = torch.utils.data.TensorDataset(X_test_, y_test_)
    train_loader = torch.utils.data.DataLoader(torch_dataset_train,shuffle=True,batch_size=12000)
    test_loader = torch.utils.data.DataLoader(torch_dataset_test,shuffle=False,batch_size=20000)

    model = Net().to(device)
    if load_True:
        model.load_state_dict(torch.load(load_), strict=False)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    ACC = 10000
    #print('11111')
    #test(model, device, train_loader)
    #test_(model, device, test_loader)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        ACC_ = test(model, device, train_loader)
        #test_(model, device, test_loader)
        if ACC_<0:
            ACC_ = -ACC_

        if ACC_<ACC or ACC_ == ACC:
            ACC = ACC_
            print(ACC)

            torch.save(model.state_dict(),save_d)
        
        scheduler.step()

if __name__ == '__main__':
    main()
