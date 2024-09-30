import numpy as np

class MPOInitialization:
    def MPOInitialization(self, max_mixed=False):
        """
        Initialize the MPO with maximally mixed state or identity.

        Parameters:
        max_mixed (bool): If True, initializes to a maximally mixed state.
        """
        self.A = np.zeros([self.chi, self.d ** 2, self.chi, self.n], dtype='complex128')  # alpha, I, alpha, modes
        self.Lambda = np.zeros([self.chi], dtype='float64')  # alpha
        self.OC = 0  # orthogonality center

        if max_mixed:
            for i in range(self.n):
                for j in range(self.d):
                    self.A[0, self.d * j + j, 0, i] = 1 / self.d
        else:
            for i in range(self.n):
                self.A[0, 0, 0, i] = 1

        self.Lambda[0] = 1