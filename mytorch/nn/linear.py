import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)

    def init_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, A):
        self.input_shape = A.shape
        batch_size = np.prod(A.shape[:-1])
        in_features = A.shape[-1]
        A_2d = A.reshape(batch_size, in_features)
        Z_2d = A_2d @ self.W.T + self.b
        out_features = self.W.shape[0]
        Z = Z_2d.reshape(*self.input_shape[:-1], out_features)
        self.A = A
        return Z

    def backward(self, dLdZ):
        out_features = self.W.shape[0]
        batch_size = np.prod(dLdZ.shape[:-1])
        dLdZ_2d = dLdZ.reshape(batch_size, out_features)
        in_features = self.W.shape[1]
        A_2d = self.A.reshape(batch_size, in_features)
        self.dLdb = np.sum(dLdZ_2d, axis=0)  
        self.dLdW = dLdZ_2d.T @ A_2d
        dLdA_2d = dLdZ_2d @ self.W
        dLdA = dLdA_2d.reshape(*self.input_shape)
        return dLdA
