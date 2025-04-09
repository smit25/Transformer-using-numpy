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
        self.Z = Z
        self.z_shifted = Z - np.max(Z, axis = self.dim, keepdims = True)
        self.z_exp = np.exp(self.z_shifted)
        self.z_sum = np.sum(self.z_exp, axis = self.dim, keepdims = True)
        self.A = self.z_exp / self.z_sum  # Done
        return self.A
        raise NotImplementedError

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            self.A = np.swapaxes(self.A, self.dim, -1)
            new_A_shape = self.A.shape
            self.A = self.A.reshape(-1, shape[-1])
            dLdA = np.swapaxes(dLdA, self.dim, -1)
            dLdA = dLdA.reshape(-1, shape[-1])
        
        N = dLdA.shape[0]  # Done
        
        dLdZ = np.zeros((N, C))  # Done

        for i in range(N):
            J = np.zeros((C,C))  # Done

            for m in range(C):
                for n in range(C):
                    J[m, n] = (self.A[i, m] * (1 - self.A[i,m]) if m == n else -self.A[i, n] * self.A[i,m])  # Done

            dLdZ[i, :] = dLdA[i, :] @ J  # Done

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = self.A.reshape(new_A_shape)
            self.A = np.swapaxes(self.A, self.dim, -1)
            dLdZ = dLdZ.reshape(self.Z.shape)
        

        return dLdZ
 

    