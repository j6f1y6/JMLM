import pandas as pd
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class MLP():
    def __init__(self) -> None:
      self.learning_rate = 0.5
      self.n_epoch = 300
      self.network = []
    
    def train(self, X, y):
        self.init_network(len(X[0]), len(y[0]))
        for idx in range(len(X)):
            self.forward(X[idx], y[idx])
        
    def init_network(self, n_feature, n_class) -> None:
        n_hidden_node = 7
        hidden_layer = [[random.random() for _ in range(n_feature)] for _ in range(n_hidden_node)]
        output_layer = [[random.random() for _ in range(n_hidden_node)] for _ in range(n_class)]
        self.network.append(hidden_layer)
        self.network.append(output_layer)

    def forward(self, x, y):
        input = x
        print(x)
        print(self.network[0][0])
        for layer in self.network:
            temp = []
            for weight in layer:
                output = input @ np.array(weight)
                print(output)
                output = self.activation(output)
                temp.append(output)
            input = np.array(temp)

    def backward(self, ):
        pass

    def activation(self, y):
        return max(0.0, y)



if __name__ == "__main__":
    data = pd.read_csv("./JMLM/datasets/Iris/Iris.csv")
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()
    y = data["Species"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    labelencoder = LabelEncoder()
    y_train = labelencoder.fit_transform(y_train).reshape(-1, 1)
    y_test = labelencoder.transform(y_test)

    onehotencoder = OneHotEncoder(sparse=False)
    y_train = onehotencoder.fit_transform(y_train)

    mlp = MLP()
    mlp.train(X_train, y_train)