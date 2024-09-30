import numpy as np

class MPO:
    def __init__(self, n, d, chi, qd, qad, seed=None):
        """
        Initialize the MPO class.

        Parameters:
        n (int): Number of qudits.
        d (int): Local dimensionality.
        chi (int): Bond dimension.
        qd (float): Depolarizing probability.
        qad (float): Amplitude damping probability.
        seed (int, optional): Random seed for reproducibility.
        """
        self.n = n
        self.d = d
        self.chi = chi
        self.qd = qd
        self.qad = qad
        self.random_state = np.random.RandomState(seed) if seed is not None else np.random