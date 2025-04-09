import numpy as np

class Softmax:
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, Z):
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        z_max = np.max(Z, axis=self.dim, keepdims=True)
        Z_stable = Z - z_max
        expZ = np.exp(Z_stable)
        sumExpZ = np.sum(expZ, axis=self.dim, keepdims=True)
        self.A = expZ / sumExpZ
        return self.A

    def backward(self, dLdA):
        dim = self.dim
        A_moved = np.moveaxis(self.A, dim, -1)      
        dLdA_moved = np.moveaxis(dLdA, dim, -1)     
        batch_size = np.prod(A_moved.shape[:-1]) 
        C = A_moved.shape[-1]
        A_2d = A_moved.reshape(batch_size, C)      
        dLdA_2d = dLdA_moved.reshape(batch_size, C) 
        temp = A_2d * dLdA_2d              
        sum_temp = np.sum(temp, axis=1, keepdims=True) 
        dLdZ_2d = temp - A_2d * sum_temp
        dLdZ_moved = dLdZ_2d.reshape(*A_moved.shape)
        dLdZ = np.moveaxis(dLdZ_moved, -1, dim)
        return dLdZ
    
