import numpy as np
from scipy.stats import unitary_group

class MPORandom_circuits:
    def RandomTwoQubitGate(self):
        """
        Generate a random two-qubit gate in the form of an MPO.

        Returns:
        np.ndarray: A random two-qubit gate represented as an MPO.
        """
        GateMPO = self.Rand_U_MPO()
        GateMPO = np.einsum('ijkl, klab -> ijab', self.noise_combined2(), GateMPO)
        return GateMPO
    
    def Rand_U(self):
        d = self.d
        return unitary_group.rvs(d**2, random_state=self.random_state).reshape(d, d, d, d)
    
    def Rand_U_MPO(self):
        """
        Generate a random unitary matrix and convert it to MPO form.

        Returns:
        np.ndarray: A random unitary matrix reshaped to MPO form.
        """
        U = self.Rand_U()
        return np.kron(U, np.conjugate(U))

    def Rand_U(self):
        """
        Generate a random unitary matrix for two qudits.

        Returns:
        np.ndarray: A random unitary matrix for two qudits.
        """
        d = self.d
        return unitary_group.rvs(d**2, random_state=self.random_state).reshape(d, d, d, d)

    def depolarizing(self):
        """
        Generate a depolarizing channel for the given dimension.

        Returns:
        np.ndarray: The depolarizing channel represented as a matrix.
        """
        d = self.d
        q = self.qd
        temp = np.zeros(d**2)
        for i in range(d):
            temp[d * i + i] = 1
        max_mixed = temp.reshape(-1, 1) @ temp.reshape(1, -1) / d

        return (1 - q) * np.identity(d ** 2) + q * max_mixed

    def depolarizing2(self):
        """
        Generate the depolarizing channel in MPO form for two qudits.

        Returns:
        np.ndarray: The depolarizing channel for two qudits.
        """
        q = self.qd
        res = self.depolarizing()
        return np.einsum('ij,kl->ikjl', res, res)

    def damping(self):
        """
        Generate an amplitude damping channel for qubits.

        Returns:
        np.ndarray: The amplitude damping channel represented as a matrix.
        """
        if self.d != 2:
            raise ValueError('Amplitude damping channel is only defined for qubits')
        q = self.qad
        return np.array([
            [1, 0, 0, q],
            [0, np.sqrt(1 - q), 0, 0],
            [0, 0, np.sqrt(1 - q), 0],
            [0, 0, 0, 1 - q]
        ])

    def damping2(self):
        """
        Generate the amplitude damping channel in MPO form for two qubits.

        Returns:
        np.ndarray: The amplitude damping channel for two qubits.
        """
        res = self.damping()
        return np.einsum('ij,kl->ikjl', res, res)

    def noise_combined(self):
        """
        Generate a noise channel combining amplitude damping and depolarizing channels.

        Returns:
        np.ndarray: The combined noise channel represented as a matrix.
        """
        return self.damping() @ self.depolarizing()

    def noise_combined2(self):
        """
        Generate the combined noise channel in MPO form for two qubits.

        Returns:
        np.ndarray: The combined noise channel for two qubits.
        """
        res = self.noise_combined()
        return np.einsum('ij,kl->ikjl', res, res)

    def RCS1DOneCycleUpdate(self, t):
        if self.OC <= self.n // 2:
            if t % 2 == 1:
                self.MPOsinglequbitUpdate(0, self.noise_combined())
            for l in range(t % 2, self.n - 1, 2):
                self.MPOtwoqubitUpdate(l, self.RandomTwoQubitGate())
            if self.OC == self.n - 3:
                self.MPOsinglequbitUpdate(self.n - 1, self.noise_combined())
            elif self.OC != self.n - 2:
                raise Exception('OC should be n-2 or n-3')
        else:
            if range(t % 2, self.n - 1, 2)[-1] != self.n - 2:
                self.MPOsinglequbitUpdate(self.n - 1, self.noise_combined())
            for l in range(t % 2, self.n - 1, 2)[::-1]:
                self.MPOtwoqubitUpdate(l, self.RandomTwoQubitGate())
            if self.OC == 1:
                self.MPOsinglequbitUpdate(0, self.noise_combined())
            elif self.OC != 0:
                raise Exception('OC should be 0 or 1')
    
    def complete_dephasing(self):
        temp = np.tensordot(self.A[:,:,:,:], np.diag(np.identity(self.d).flatten()), axes = ([1], [0])) # output: alpha, alpha, modes, I 
        self.Gamma[:,:,:,:] = np.transpose(temp, (0, 3, 1, 2))