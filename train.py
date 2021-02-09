import sys
import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":

    np.random.seed(7)

    try:
        df = pd.read_csv(sys.argv[1], header=None)
        df = df.drop(df.columns[0], axis=1)
    except:
        exit("Error: Something went wrong with the dataset")

    label = df.iloc[:, 0].tolist()

    Y = np.array([np.array([1,0]) if x == 'M' else np.array([0,1]) for x in label])
    df = df.iloc[:, 1:]

    X = pd.get_dummies(df).to_numpy()

    model = NeuralNetwork()

    model.add_layer(500, input_dim=X.shape[1], activation='sigmoid')
    model.add_layer(100, activation='sigmoid')
    model.add_layer(50, activation='sigmoid')
    model.add_layer(2, activation='softmax')

    model.summary()

    model.fit(X, Y, epoch=200, verbose=0, normalize=True, lr=0.6)
    model.save()
