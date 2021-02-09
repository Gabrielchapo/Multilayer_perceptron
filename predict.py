import sys
import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":

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

    model.load()
    prediction = model.predict(X, normalize=True)
    
    prediction = np.argmax(prediction, axis=1)
    real = np.argmax(Y, axis=1)
    pred_test= model.predict(X)

    count = 0

    for i in range(len(prediction)):
        if prediction[i] == real[i]:
            count += 1

    print("Accuracy: ", count / len(prediction))

