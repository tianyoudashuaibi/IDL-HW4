import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    def __init__(self):
        self.eps = 1e10
        self.softmax = Softmax(dim=-1)
        
    def forward(self, Q, K, V, mask=None):
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        d_k = Q.shape[-1] 
        K_t = np.swapaxes(K, -1, -2)   
        scaled_dot_product = np.matmul(Q, K_t) / np.sqrt(d_k)
        if mask is not None:
            scaled_dot_product[mask] = -self.eps
        self.attention_scores = self.softmax.forward(scaled_dot_product)  
        output = np.matmul(self.attention_scores, V)
        return output
    
    def backward(self, d_output):
        Q, K, V = self.Q, self.K, self.V
        A = self.attention_scores      
        d_k = Q.shape[-1]               
        A_t = np.swapaxes(A, -1, -2)     
        d_V = np.matmul(A_t, d_output)  
        V_t = np.swapaxes(V, -1, -2)     
        dA = np.matmul(d_output, V_t)  
        dScaled = self.softmax.backward(dA)  
        if self.mask is not None:
            dScaled[self.mask] = 0.
        dScaled = dScaled / np.sqrt(d_k)
        d_Q = np.matmul(dScaled, K) 
        dScaled_t = np.swapaxes(dScaled, -1, -2)  
        dK_temp = np.matmul(dScaled_t, Q)         
        d_K = dK_temp
        return d_Q, d_K, d_V
