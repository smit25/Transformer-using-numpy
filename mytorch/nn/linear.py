import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        self.input_shape = A.shape
        A = A.reshape(-1, self.input_shape[-1])
        Z = A@self.W.T + self.b
        Z = Z.reshape(self.input_shape[:-1] + (self.b.shape[0],))
        
        # Store input for backward pass
        self.A = A
        return Z
        
        raise NotImplementedError

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        dLdZ = dLdZ.reshape(-1, dLdZ.shape[-1])

        # Compute gradients (refer to the equations in the writeup)
        self.dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = np.sum(dLdZ, axis =0)
        self.dLdA = self.dLdA.reshape(self.input_shape)
        
        # Return gradient of loss wrt input
        return self.dLdA
        raise NotImplementedError
