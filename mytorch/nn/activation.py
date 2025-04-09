import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        z_max = np.max(Z, axis=self.dim, keepdims=True)
        Z_stable = Z - z_max
        expZ = np.exp(Z_stable)
        sumExpZ = np.sum(expZ, axis=self.dim, keepdims=True)
        self.A = expZ / sumExpZ
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        shape = self.A.shape
        # Dimension along which we applied softmax
        dim = self.dim
        
        # 1) Move the softmax dimension to the last axis for convenience
        #    If dim is negative, np.moveaxis automatically handles it,
        #    but we can also do dim % len(shape) if needed.
        A_moved = np.moveaxis(self.A, dim, -1)       # shape = (..., C)
        dLdA_moved = np.moveaxis(dLdA, dim, -1)      # shape = (..., C)
        
        # 2) Flatten all leading dimensions into one "batch" dimension
        #    so we get a 2D shape of (batch_size, C)
        batch_size = np.prod(A_moved.shape[:-1])     # product of everything except last axis
        C = A_moved.shape[-1]
        
        A_2d = A_moved.reshape(batch_size, C)        # shape = (batch_size, C)
        dLdA_2d = dLdA_moved.reshape(batch_size, C)  # shape = (batch_size, C)
        
        # 3) Standard 2D softmax backward
        #    derivative of softmax: dLdZ = A * (dLdA - sum_over_C(A*dLdA))
        temp = A_2d * dLdA_2d              # elementwise
        sum_temp = np.sum(temp, axis=1, keepdims=True)  # shape = (batch_size, 1)
        dLdZ_2d = temp - A_2d * sum_temp
        
        # 4) Reshape back to the original "moved" shape, then move axis back
        dLdZ_moved = dLdZ_2d.reshape(*A_moved.shape)
        dLdZ = np.moveaxis(dLdZ_moved, -1, dim)
        
        return dLdZ
 

    
    
