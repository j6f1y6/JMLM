import numpy as np
import torchvision
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import time
from torch import nn
from load_data import load_data
 
class MLP(nn.Module):

    def __init__(self, n_input=28*28, n_hidden=100, n_output=10):
        super(MLP,self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
  
    def forward(self, x):
        return self.linear_relu_stack
 

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data("mnist")

    print(len(X_train))
    mlp=MLP()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters())
    epochs = 5
    batch_size = 64
    
    print(X_train.shape)
    train_idx = np.arange(0, len(X_train), batch_size)

    for epoch in range(epochs) :
        np.random.shuffle(train_idx)
        sum_loss = 0
        train_correct = 0
        for batch_idx in train_idx:
            inputs, labels = X_train[batch_idx:batch_idx+batch_size], y_train[batch_idx:batch_idx+batch_size] #inputs 维度：[64,1,28,28]
            # print(inputs.shape)
            # inputs=torch.flatten(inputs,start_dim=1) #展平数据，转化为[64,784]
        #     print(inputs.shape)
    #         outputs=model(inputs)
    #         optimizer.zero_grad()
    #         loss=cost(outputs,labels)
    #         loss.backward()
    #         optimizer.step()
    
    #         _,id=torch.max(outputs.data,1)
    #         sum_loss+=loss.data
    #         train_correct+=torch.sum(id==labels.data)
    #     print('[%d,%d] loss:%.03f' % (epoch + 1, epochs, sum_loss / len(data_loader_train)))
    #     print('        correct:%.03f%%' % (100 * train_correct / len(data_train)))
    #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    # model.eval()
    # test_correct = 0
    # for data in data_loader_test :
    #     inputs, lables = data
    #     inputs, lables = Variable(inputs).cpu(), Variable(lables).cpu()
    #     inputs=torch.flatten(inputs,start_dim=1) #展并数据
    #     outputs = model(inputs)
    #     _, id = torch.max(outputs.data, 1)
    #     test_correct += torch.sum(id == lables.data)
    # print("correct:%.3f%%" % (100 * test_correct / len(data_test )))