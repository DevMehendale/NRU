import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import pandas as pd
from sklearn.model_selection import train_test_split

import warnings

from torch.nn.utils import clip_grad_norm_
import pickle

import torch
warnings.filterwarnings('ignore')

data = pd.read_csv('Mnist/mnist_train.csv')
data_test = pd.read_csv('Mnist/mnist_test.csv')

data = data.sample(frac=1,random_state=13,axis=1).reset_index(drop=True)    #index reset, if not done, was throwing an error on kubernetes cluster

Y=np.array(data['label'])
X=np.array(data.drop(['label'],axis=1))
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=0)
print(X_train.shape, X_val.shape)

X_train, Y_train, X_val, Y_val = map(torch.tensor, (X_train, Y_train, X_val, Y_val))
print(X_train.shape, Y_train.shape)

class NRUcell(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, memory_dim, num_heads,use_relu=False,_layer_norm=False):
        super(NRUcell,self).__init__()
        self.hidden_dim=hidden_dim
        self.input_dim=input_dim
        self.memory_dim=memory_dim
        self.num_heads=num_heads
        
        
        self.fc_v_alpha=torch.nn.Linear(hidden_dim+memory_dim, 2*int(np.sqrt(num_heads*memory_dim)))
        self.fc_v_beta=torch.nn.Linear(hidden_dim+memory_dim, 2*int(np.sqrt(num_heads*memory_dim)))
        
        self.fc_alpha=torch.nn.Linear(hidden_dim+memory_dim, num_heads)
        self.fc_beta=torch.nn.Linear(hidden_dim+memory_dim, num_heads)
        
        self.r=torch.nn.ReLU()
        self._layer_norm=_layer_norm
        
        self.layernorm = torch.nn.LayerNorm(self.hidden_dim)
        
        self.fc_to_h = torch.nn.Linear(memory_dim + hidden_dim + input_dim, hidden_dim)
        self.use_relu=use_relu
        
    def _relu(self, x):
        if self.use_relu:
            return self.r(x)
        else:
            return x
        
    def _layernorm(self, x):
        if self._layer_norm:
            return self.layernorm(x)
        else:
            return x
    
    def forward(self, x, h, m, i):
        h=self.r(self._layernorm(self.fc_to_h(torch.cat((h,torch.reshape(x[:,i,:],(x.size(0),input_dim)),m),1))))                
        input1=torch.cat((torch.reshape(h,(x.size(0),-1)),torch.reshape(m,(x.size(0),-1))),1)

        alpha = self._relu(self.fc_alpha(input1)).clone()
        beta = self._relu(self.fc_beta(input1)).clone()

        u_alpha = self.fc_v_alpha(input1).chunk(2,dim=1)
        v_alpha = torch.bmm(u_alpha[0].unsqueeze(2), u_alpha[1].unsqueeze(1)).view(-1, self.num_heads, self.memory_dim)
        v_alpha = self._relu(v_alpha)
        v_alpha = torch.nn.functional.normalize(v_alpha, p=5, dim=2, eps=1e-12)
        add_memory = alpha.unsqueeze(2)*v_alpha

        u_beta = self.fc_v_beta(input1).chunk(2, dim=1)
        v_beta = torch.bmm(u_beta[0].unsqueeze(2), u_beta[1].unsqueeze(1)).view(-1, self.num_heads, self.memory_dim)
        v_beta = self._relu(v_beta)
        v_beta = torch.nn.functional.normalize(v_beta, p=5, dim=2, eps=1e-12)
        forget_memory = beta.unsqueeze(2)*v_beta

        m = m + torch.mean(add_memory-forget_memory, dim=1)
        return h,m
            
            
class NRUModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_dim, batch_size, output_dim,num_heads,use_relu=False,_layer_norm=False):
        super(NRUModel,self).__init__()
        self.hidden_dim = hidden_dim

        self.batch_size = batch_size
        self.memory_dim = memory_dim
        self._layer_norm=_layer_norm

        self.nru = NRUcell(input_dim, hidden_dim, memory_dim,num_heads,use_relu,_layer_norm)

        self.h0 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty( self.batch_size, self.hidden_dim)).requires_grad_())
        self.m0 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty( self.batch_size, self.memory_dim)).requires_grad_())
        
        self._W_h2o = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(hidden_dim, output_dim)))
        self._b_o = torch.nn.Parameter(torch.nn.init.constant_(torch.Tensor(output_dim),0))
        
    def forward(self, x):
        x=x.cuda()
        
        for i in range(x.size(1)):
            if i==0:
                h,m = self.nru(x, self.h0, self.m0, i)
            else:
                h,m = self.nru(x, h, m, i)
    
        out = torch.add(torch.mm(h, self._W_h2o), self._b_o)
        return out


input_dim=1
hidden_dim=212
memory_dim=256  
num_heads=4     
output_dim=10
batch_size=100
seq_dim = 784
clip_norm=1
lr=0.001
use_relu=False
_layer_norm=False
rseed=5

torch.manual_seed(rseed)


model=NRUModel(input_dim, hidden_dim, memory_dim, batch_size, output_dim,num_heads)
model.cuda()

loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer= torch.optim.Adam(model.parameters(), lr=lr)


X_train=X_train.float()
X_val=X_val.float()
Y_train=Y_train
Y_val=Y_val

def loss_fn(out,y):
    loss = loss_func(out, y.cuda())
    return loss

def evaluate(model,x,y):
    model.eval()
    correct = 0.0
    total = 0.0
    for i in range(0,x.size()[0], batch_size):
        inputs=torch.reshape(x[i:i+batch_size],(-1,seq_dim, input_dim))

        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        total += y[i:i+batch_size].size(0)

        correct += (predicted.cpu() == y[i:i+batch_size]).sum()
    accuracy=100 * correct / total
    return accuracy

def step(model,inputs,y):
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_fn(outputs, y)

        loss.backward(retain_graph=False)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, norm_type=2)

        optimizer.step()
        return loss

epochs=100
iteration=0
acc_train=[]
acc_val=[]
loss_train=[]
for epoch in range(epochs):
    permutation = torch.randperm(X_train.size()[0])
    model.train()
    for i in range(0,100, batch_size):
        inputs=torch.reshape(X_train[permutation[i:i+batch_size]],(-1,seq_dim, input_dim))

        loss=step(model,inputs,Y_train[permutation[i:i+batch_size]])
        
        iteration=iteration+1
        
        if iteration%1==0:
            print(epoch, iteration, loss.item())
        
    torch.cuda.empty_cache()
    
    
    accuracy1=evaluate(model,X_train,Y_train)
    accuracy2=evaluate(model,X_val,Y_val)
    
    acc_train.append(accuracy1)
    acc_val.append(accuracy2)
    loss_train.append(loss.item())
    
    print(epoch, iteration, loss.item(), accuracy1.item(), accuracy2.item())
    iteration=0
    
    

mylist=[acc_train,acc_val,loss_train]
with open('nru_graph.pkl', 'wb') as f:
    pickle.dump(mylist, f)


torch.save(model.state_dict(), 'nru.pt')

