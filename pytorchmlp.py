import torch
import numpy as np
import math
import tqdm
from torch import nn
import pandas as pd
from jmlm import JMLM
from load_data import load_data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, internal_dim=16) -> None:
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            # nn.Conv2d(3, 10, 3, 2, ),
            # nn.Tanh(),
            # nn.Conv2d(10, 20, 3, 2),
            # nn.Tanh(),
            nn.Flatten(),
            nn.Linear(input_dim, internal_dim),
            nn.Tanh(),
            nn.Linear(internal_dim, internal_dim),
            nn.Tanh(),
            nn.Linear(internal_dim, internal_dim),
            nn.Tanh(),
        )

        self.output = nn.Sequential(
            nn.Linear(internal_dim, output_dim), 
            # nn.Softmax(dim=1),    
        )
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return self.output(logits)



if __name__ == '__main__':
    epoch = 1000
    learning_rate = 0.001
    batch_size = 16
    mlp_internal_dim = 16
    X_train, X_test, y_train, y_test = load_data("cifar10")

    # labelencoder = LabelEncoder()
    # y_train = labelencoder.fit_transform(y_train).reshape(-1, 1)
    # y_test = labelencoder.transform(y_test)

    # onehotencoder = OneHotEncoder()
    # y_train=onehotencoder.fit_transform(y_train).toarray()
    
    input_dim = math.prod(X_train.shape[1:])
    mlp = MLP(input_dim, y_train.shape[1], mlp_internal_dim).cuda()
    
    optimizor = torch.optim.Adam(params= mlp.parameters(), lr=learning_rate)
    loss = torch.nn.CrossEntropyLoss()

    train_idx = np.arange(0, len(X_train), batch_size)

    # X_train, X_test, y_train, y_test = torch.from_numpy(X_train).transpose(1, 3).float().cuda(), \
    #                                 torch.from_numpy(X_test).transpose(1, 3).float().cuda(), \
    #                                 torch.from_numpy(y_train).float().cuda(),\
    #                                 torch.from_numpy(y_test).float().cuda()
    X_train, X_test, y_train, y_test = torch.from_numpy(X_train).float().cuda(), \
                                    torch.from_numpy(X_test).float().cuda(), \
                                    torch.from_numpy(y_train).float().cuda(),\
                                    torch.from_numpy(y_test).float().cuda()

    for _ in tqdm.tqdm(range(epoch)):
        np.random.shuffle(train_idx)
        for bi in train_idx:
            optimizor.zero_grad()
            bx, by = X_train[bi:bi+batch_size], y_train[bi:bi+batch_size]
            by_pred = mlp(bx)
            err = loss(by_pred, by)
            err.backward()
            optimizor.step()

    predict_out = mlp(X_test)
    _, predict_y = torch.max(predict_out, 1)

    print('Prediction Accuracy in MLP:', accuracy_score(y_test.cpu(), predict_y.cpu()))

    new_X_train = mlp.linear_relu_stack(X_train).detach().cpu().numpy()
    new_X_test = mlp.linear_relu_stack(X_test).detach().cpu().numpy()


    jmlm = JMLM(64, 10)
    jmlm.train(new_X_train, y_train.detach().cpu().numpy(), 5, 20)
    y_pred = jmlm.predict(new_X_test)
    acc = accuracy_score(y_test.detach().cpu().numpy(), y_pred)
    # y_pred=onehotencoder.transform(y_pred.reshape(-1, 1)).toarray()
    print('Prediction Accuracy after JMLM:', acc)