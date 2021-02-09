import numpy as np
from matplotlib import pyplot as plt

def sigmoid_derv(s):
    return s * (1 - s)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def error(Y_pred, Y_real):
    logged = - np.log(Y_pred[np.arange(Y_real.shape[0]), Y_real.argmax(axis=1)])
    return np.sum(logged) / Y_real.shape[0]

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.nb_layers = 0

    def add_layer(self, size, activation, input_dim=None):

        layer = {}
        if self.nb_layers == 0 and input_dim is None:
            exit("First layer needs input dimension")

        if input_dim is None:
            input_dim = self.layers[-1]['weights'].shape[1]

        layer["name"] = "layer_" + str(self.nb_layers)
        layer['activation'] = activation
        layer['weights'] = np.random.randn(input_dim, size)
        layer['bias'] = np.zeros((1, size))

        self.layers.append(layer)
        self.nb_layers += 1

    def summary(self):
        for x in self.layers:
            print("Layer:", x["name"], "| Dimensions:", x["weights"].shape, "| Activation:", x["activation"])

    def save(self):
        dict = {"layers":self.layers, "mean":self.mean, "sigma":self.sigma}
        try:
            np.save("resources/model.npy", dict)
        except:
            exit("Something went wrong while saving model.")

    def load(self):
        try:
            dict = np.load("resources/model.npy", allow_pickle='TRUE').item()
        except:
            exit("Something went wrong while loading model.")
        self.sigma = dict["sigma"]
        self.mean = dict["mean"]
        self.layers = dict["layers"]

    def fit(self, X, Y, epoch, verbose=False, normalize=False, batch_size=None, lr=0.05):

        assert epoch > 0

        self.lr = lr

        if normalize is True:
            self.sigma = [np.amax(x) - np.amin(x) if np.amax(x) - np.amin(x) != 0 else 1 for x in zip(*X)]
            self.mean = [sum(x) / len(X) for x in zip(*X)]
            X = (X - self.mean) / self.sigma

        val_size = round(X.shape[0] * 0.3)

        X_val = X[:val_size]
        Y_val = Y[:val_size]
        X_train = X[val_size:]
        Y_train = Y[val_size:]
        
        if batch_size is None:
            batch_size = X_train.shape[0]
        else:
            assert batch_size <= X_train.shape[0]
        
        self.activated = [None] * self.nb_layers
        self.activated_delta = [None] * self.nb_layers

        list_err_train = []
        list_err_val = []

        for i in range(epoch):
            for x, y in zip(batch(X_train, batch_size), batch(Y_train, batch_size)):
                self.__feedforward(x)
                self.__backprop(x,y)
            if verbose:
                self.__feedforward(X)
                err_train = error(self.activated[-1], Y_train)
                list_err_train.append(err_train)
                err_val = error(self.activated[-1], Y_val)
                list_err_val.append(err_val)
                print("epoch ", i + 1, "/", epoch, " - loss: ", err_train,  " - val_loss: ", err_val,sep="")
        if verbose:
            plt.plot(list_err_train, label="loss")
            plt.plot(list_err_val, label="val_loss")
            plt.legend()
            plt.show()


    def __feedforward(self, X):
        
        tmp = X
        for i, layer in enumerate(self.layers):
            z = np.dot(tmp, layer["weights"]) + layer["bias"]
            self.activated[i] = eval(layer["activation"])(z)
            tmp = self.activated[i]

    def __backprop(self, X, Y):

        for i in reversed(range(self.nb_layers)):
            if i == self.nb_layers - 1:
                self.activated_delta[i] = (self.activated[i] - Y) / Y.shape[0]
            else:
                self.activated_delta[i] = np.dot(self.activated_delta[i+1], self.layers[i+1]["weights"].T) * sigmoid_derv(self.activated[i])

        for i in reversed(range(self.nb_layers)):
            if i != 0:
                self.layers[i]["weights"] -= self.lr * np.dot(self.activated[i-1].T, self.activated_delta[i])
            else:
                self.layers[i]["weights"] -= self.lr * np.dot(X.T, self.activated_delta[i])
            if i == self.nb_layers - 1:
                self.layers[i]["bias"] -= self.lr * np.sum(self.activated_delta[i], axis=0, keepdims=True)
            else:
                self.layers[i]["bias"] -= self.lr * np.sum(self.activated_delta[i], axis=0)

    def predict(self, X, normalize=False):
        if normalize is True:
            X = (X - self.mean) / self.sigma
        self.activated = [None] * len(self.layers)
        self.__feedforward(X)
        return self.activated[-1].tolist()
