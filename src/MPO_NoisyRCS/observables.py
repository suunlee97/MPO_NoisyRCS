import numpy as np

def EntropyFromRow(InputRow):
    Output = 0

    for i in range(len(InputRow)):
        if InputRow[0, i] == 0:
            Output = Output
        else:
            Output = Output - InputRow[0,i] * np.log2(InputRow[0,i])

    return Output

def EntropyFromColumn(InputColumn):
    Output = np.nansum(-InputColumn * np.log2(InputColumn))
    
    return Output

def ColumnSumToOne(InputColumn):
    return InputColumn / np.sum(InputColumn)

class MPOObservables:
    def getprobs(self):
        """
        Compute the output distribution from the MPO.
        """
        l = self.OC

        if not self.d == 2:
            raise ValueError('getprobs is only defined for qubits')

        temp = self.A[0, (0, 3), :, 0] # this kind of indexing maps (0,3) to the first index        
        for i in range(self.n - 1):
            if i == l:
                temp = np.tensordot(temp, np.diag(self.Lambda), axes = (-1, 0))
            if i < self.n-2:
                temp = np.tensordot(temp, np.transpose(self.A[:, (0, 3), :, i + 1], (1, 0, 2)), axes = (-1, 0)) # this kind of indexing maps (0,3) to the first index and thus needs to be transposed
            else:
                temp = np.tensordot(temp, self.A[:, (0, 3), 0, i + 1], axes = (-1, 0)) # strange! but this kind of indexing does *not* (0,3) to the first index and thus needs to be transposed

        probs = np.real(temp.reshape(-1))

        return probs

    def getprobs_partial(self, chunk_index, num_output_bits):
        """
        Compute the output distribution from the MPO for a bitstrings with fixed first (n - num_output_bits) bits.
        """
        if not self.d == 2:
            raise ValueError('getprobs is only defined for qubits')
        if num_output_bits > self.n:
            raise ValueError('num_output_bits must be less than or equal to the number of qubits in the MPO')
        if chunk_index >= 2 ** (self.n - num_output_bits):
            raise ValueError('chunk_index must be less than 2^(n-num_output_bits)')
        
        num_fixed_bits = self.n - num_output_bits
        # fixed_bits = format(chunk_index, f'0{num_fixed_bits}b')
        fixed_bits = format(chunk_index, f'0{num_fixed_bits}b')
        fixed_bits_list = np.array([int(bit) for bit in fixed_bits])

        l = self.OC

        if fixed_bits_list[0] == 0:
            temp = self.A[0, 0, :, 0]
        else:
            temp = self.A[0, 3, :, 0]
        
        for i in range(num_fixed_bits-1):
            if i == l:
                temp = np.tensordot(temp, np.diag(self.Lambda), axes = (-1, 0))
            if fixed_bits_list[i + 1] == 0:
                temp = np.tensordot(temp, self.A[:, 0, :, i + 1], axes = (-1, 0))
            else:
                temp = np.tensordot(temp, self.A[:, 3, :, i + 1], axes = (-1, 0))

        for i in range(num_fixed_bits-1, self.n-1):
            if i == l:
                temp = np.tensordot(temp, np.diag(self.Lambda), axes = (-1, 0))
            if i < self.n-2:
                temp = np.tensordot(temp, np.transpose(self.A[:, (0, 3), :, i + 1], (1, 0, 2)), axes = (-1, 0)) # this kind of indexing maps (0,3) to the first index and thus needs to be transposed
            else:
                temp = np.tensordot(temp, self.A[:, (0, 3), 0, i + 1], axes = (-1, 0)) # strange! but this kind of indexing does *not* (0,3) to the first index and
        
        probs_partial = np.real(temp.reshape(-1))

        return probs_partial


    def TotalProbFromMPO(self):
        """
        Calculate the total probability from the MPO.
        """
        l = self.OC
        A_reshaped = self.A[0, :, :, 0].reshape(self.d, self.d, self.chi)
        temp = np.trace(A_reshaped, axis1=0, axis2=1)

        for i in range(self.n - 1):
            if i == l:
                temp = temp * self.Lambda
            A_reshaped = self.A[:, :, :, i + 1].reshape(self.chi, self.d, self.d, self.chi)
            temp = np.tensordot(temp, np.trace(A_reshaped, axis1=1, axis2=2), axes=(0, 0))

        return temp[0]
    
    def TotalPurityFromMPO(self):
        l = self.OC
        # Initial tensor contraction for the first Gamma tensor
        A_reshaped = self.A[0, :, :, 0].reshape(self.d, self.d, self.chi)
        temp = np.tensordot(A_reshaped, A_reshaped, axes=([0, 1], [1, 0]))  # temp[alpha, alpha']

        for i in range(self.n - 1):
            if i == l:
                Lambda_double = np.outer(self.Lambda[:], self.Lambda[:])
                temp *= Lambda_double

            # Reshape Gamma tensors for the current site
            A_reshaped = self.A[:, :, :, i+1].reshape(self.chi, self.d, self.d, self.chi) # Gamma_reshaped[alpha,beta,i,j]

            temp = np.tensordot(
                np.tensordot(temp, A_reshaped, axes=([0], [0])), # temp[alpha,alpha'] Gamma_reshaped[alpha,beta,i,j] = temp[alpha',beta,i,j]
                A_reshaped,
                axes=([0, 1, 2], [0, 2, 1]) # temp[alpha',beta,i,j] Gamma_reshaped[alpha',beta',j,i] = temp[beta,beta']
            )

        return temp[0, 0]

    def MarginalPurityFromMPO(self, marginal):
        
        if not set(marginal).issubset(range(self.n)):
            raise ValueError("marginal must be a subset of range(self.n)")
        
        l = self.OC
        # Initial tensor contraction for the first Gamma tensor
        A_reshaped = self.A[0, :, :, 0].reshape(self.d, self.d, self.chi)
        if 0 in marginal:
            temp = np.tensordot(A_reshaped, A_reshaped, axes=([0, 1], [1, 0]))  # temp[alpha, alpha']
        else:
            A_reshaped_tr = np.trace(A_reshaped, axis1=0, axis2=1)
            temp = np.outer(A_reshaped_tr,
                            A_reshaped_tr)
        for i in range(self.n - 1):
            if i == l:
                Lambda_double = np.outer(self.Lambda[:], self.Lambda[:])
                temp *= Lambda_double
            
            # Reshape Gamma tensors for the current site
            A_reshaped = self.A[:, :, :, i+1].reshape(self.chi, self.d, self.d, self.chi)
            
            # Update temp with Gamma
            if i+1 in marginal:
                temp = np.tensordot(
                    np.tensordot(temp, A_reshaped, axes=([0], [0])), # temp[alpha,alpha'] Gamma_reshaped[alpha,beta,i,j] = temp[alpha',beta,i,j]
                    A_reshaped, axes=([0, 1, 2], [0, 2, 1]) # temp[alpha',beta,i,j] Gamma_reshaped[alpha',beta',j,i] = temp[beta,beta']
                ) 
            else:
                A_reshaped_tr = np.trace(A_reshaped, axis1=1, axis2=2)
                temp = np.tensordot(
                    np.tensordot(temp, A_reshaped_tr, axes=([0],[0])), # temp[alpha,alpha'] trGamma_reshaped[alpha,beta] = temp[alpha',beta]
                    A_reshaped_tr, axes=([0],[0]) # temp[alpha',beta] trGamma_reshaped[alpha',beta'] = temp[beta,beta']
                ) 

        return temp[0, 0]
    
    def MPOEntanglementEntropy(self):
        """
        Calculate the entanglement entropy across different bipartitions.
        """
        Output = np.zeros([self.n - 1])

        # Move the orthogonality center to the leftmost or rightmost side, whichever is nearer
        if self.OC <= self.n // 2:
            while self.OC > 0:
                self.MPOMoveOC('left')
            sq_lambda = np.copy(self.Lambda ** 2)
            Output[0] = EntropyFromColumn(ColumnSumToOne(sq_lambda))
            for i in range(1, self.n - 1):
                self.MPOMoveOC('right')
                sq_lambda = np.copy(self.Lambda ** 2)
                Output[i] = EntropyFromColumn(ColumnSumToOne(sq_lambda))
        else:
            while self.OC < self.n - 2:
                self.MPOMoveOC('right')
            sq_lambda = np.copy(self.Lambda ** 2)
            Output[-1] = EntropyFromColumn(ColumnSumToOne(sq_lambda))
            for i in range(1, self.n - 1):
                self.MPOMoveOC('left')
                sq_lambda = np.copy(self.Lambda ** 2)
                Output[-1 - i] = EntropyFromColumn(ColumnSumToOne(sq_lambda))

        return Output
    
    def compute_CMI(self, x):
        """
        Compute the Conditional Mutual Information (CMI) from the MPO.

        Parameters:
        x (int): The number of qudits defining the subsystem.

        A: left n/2 - x qudits
        B: middle 2x qudits
        C: right n/2 - x qudits

        Returns:
        float: The computed CMI.
        """
        qubitnum = self.n
        purity_AB = self.MarginalPurityFromMPO(range(qubitnum // 2 + x))
        purity_BC = self.MarginalPurityFromMPO(range(qubitnum // 2 - x, qubitnum))
        purity_B = self.MarginalPurityFromMPO(range(qubitnum // 2 - x, qubitnum // 2 + x))
        purity_ABC = self.TotalPurityFromMPO()
        return - np.log2(purity_AB.real) - np.log2(purity_BC.real) + np.log2(purity_B.real) + np.log2(purity_ABC.real)
    
    def compute_MI(self, x):
        """
        Compute the Mutual Information (MI) from the MPO.

        Parameters:
        x (int): the index of the cut

        A: left x+1 qudits
        B: right n - x - 1 qudits

        Returns:
        float: The computed MI.
        """
        qubitnum = self.n
        purity_A = self.MarginalPurityFromMPO(range(x+1))
        purity_B = self.MarginalPurityFromMPO(range(x+1, qubitnum))
        purity_AB = self.TotalPurityFromMPO()
        return - np.log2(purity_A.real) - np.log2(purity_B.real) + np.log2(purity_AB.real)
    
    def page_MI(self):
        """
        Compute the Page curve of Mutual Information (MI) from the MPO.

        Returns:
        array: .
        """
        n = self.n
        Output = np.zeros(n-1)
        for x in range(n-1):
            Output[x] = self.compute_MI(x)
        return Output

    def coll_prob(self):
        l = self.OC

        # Copy A not to affect the original MPO
        A_temp = np.copy(self.A)
        
        # Apply complete dephasing to the copied A
        temp = np.tensordot(A_temp[:,:,:,:], np.diag(np.identity(self.d).flatten()), axes = ([1], [0])) # output: alpha, I, alpha, modes 
        A_temp[:,:,:,:] = np.transpose(temp, (0, 3, 1, 2))
        
        # Compute the purity of the dephased MPO

        # Initial tensor contraction for the first A tensor
        A_reshaped = A_temp[0, :, :, 0].reshape(self.d, self.d, self.chi)
        temp = np.tensordot(A_reshaped, A_reshaped, axes=([0, 1], [1, 0]))  # temp[alpha, alpha']

        for i in range(self.n - 1):
            if i == l:
                Lambda_double = np.outer(self.Lambda[:], self.Lambda[:])
                temp *= Lambda_double

            # Reshape A tensors for the current site
            A_reshaped = A_temp[:, :, :, i+1].reshape(self.chi, self.d, self.d, self.chi) # A_reshaped[alpha,beta,i,j]

            temp = np.tensordot(
                np.tensordot(temp, A_reshaped, axes=([0], [0])), # temp[alpha,alpha'] A_reshaped[alpha,beta,i,j] = temp[alpha',beta,i,j]
                A_reshaped,
                axes=([0, 1, 2], [0, 2, 1]) # temp[alpha',beta,i,j] A_reshaped[alpha',beta',j,i] = temp[beta,beta']
            )

        Purity = temp[0, 0]

        # Normalize and return
        return (2 ** self.n) * Purity.real
    
    @staticmethod
    def Inner_product(mpo1,mpo2):
        """
        Compute the inner product of two MPOs: <rho_1,rho_2> = Tr(rho_1^{\dagger} rho_2).
        """

        if mpo1.n != mpo2.n:
            raise ValueError("The two MPOs must have the same number of qudits.")
        if mpo1.d != mpo2.d:
            raise ValueError("The two MPOs must have the same local dimension.")
        if mpo1.chi != mpo2.chi:
            raise ValueError("The two MPOs must have the same bond dimension.")
        
        n = mpo1.n
        d = mpo1.d
        chi = mpo1.chi
        
        l1 = mpo1.OC
        l2 = mpo2.OC

        # Initial tensor contraction for the first Gamma tensor
        A1_c = np.conjugate(mpo1.A[0, :, :, 0])
        A2 = mpo2.A[0, :, :, 0]
        temp = np.tensordot(A1_c, A2, axes=([0], [0])) # temp[alpha, alpha']

        for i in range(n - 1):
            if i == l1:
                temp = np.tensordot(temp, np.diag(mpo1.Lambda), axes=([0], [0])) # temp[alpha,alpha'] Lambda[alpha,beta] = temp[alpha',beta]
                temp = np.transpose(temp, (1, 0)) # temp[beta,alpha']
            if i == l2:
                temp = np.tensordot(temp, np.diag(mpo2.Lambda), axes=([1], [0])) # temp[alpha,alpha'] Lambda[alpha',beta'] = temp[alpha,beta']

            # Reshape Gamma tensors for the current site
            A1_c = np.conjugate(mpo1.A[:, :, :, i+1]) # A1_c[alpha,I,beta]
            A2 = mpo2.A[:, :, :, i+1] # A2[alpha,I,beta]

            temp = np.tensordot(
                np.tensordot(temp, A1_c, axes=([0], [0])), # temp[alpha,alpha'] A1_c[alpha,I,beta] = temp[alpha',I,beta]
                    A2,
                    axes=([0, 1], [0, 1]) # temp[alpha',I,beta] A2[alpha',I,beta'] = temp[beta,beta']
            )

        return temp[0, 0]
    
    @staticmethod
    def Inner_product_prob(mpo1, mpo2):
        """
        Compute the inner product of two MPOs after complete dephasing: <rho_1,rho_2> = Tr(rho_1^{\dagger} rho_2).
        """
        # raise NotImplementedError("Inner product after complete dephasing needs to be debugged.")
    
        if mpo1.n != mpo2.n:
            raise ValueError("The two MPOs must have the same number of qudits.")
        if mpo1.d != mpo2.d:
            raise ValueError("The two MPOs must have the same local dimension.")
        if mpo1.chi != mpo2.chi:
            raise ValueError("The two MPOs must have the same bond dimension.")
        
        n = mpo1.n
        d = mpo1.d
        chi = mpo1.chi
        
        l1 = mpo1.OC
        l2 = mpo2.OC

        diag_indices = d * np.arange(d) + np.arange(d)

        # Initial tensor contraction for the first Gamma tensor
        A1_c = np.conjugate(mpo1.A[0, :, :, 0])
        A1_c = A1_c[diag_indices, :]

        A2 = mpo2.A[0, :, :, 0]
        A2 = A2[diag_indices, :]

        temp = np.tensordot(A1_c, A2, axes=([0], [0])) # temp[alpha, alpha']

        for i in range(n - 1):
            if i == l1:
                temp = np.tensordot(temp, np.diag(mpo1.Lambda), axes=([0], [0])) # temp[alpha,alpha'] Lambda[alpha,beta] = temp[alpha',beta]
                temp = np.transpose(temp, (1, 0)) # temp[beta,alpha']
            if i == l2:
                temp = np.tensordot(temp, np.diag(mpo2.Lambda), axes=([1], [0])) # temp[alpha,alpha'] Lambda[alpha',beta'] = temp[alpha,beta']

            # Reshape Gamma tensors for the current site
            A1_c = np.conjugate(mpo1.A[:, :, :, i+1]) # A1_c[alpha,I,beta]
            A1_c = A1_c[:, diag_indices, :]
            
            A2 = mpo2.A[:, :, :, i+1] # A2[alpha,I,beta]
            A2 = A2[:, diag_indices, :]

            temp = np.tensordot(
                np.tensordot(temp, A1_c, axes=([0], [0])), # temp[alpha,alpha'] A1_c[alpha,I,beta] = temp[alpha',I,beta]
                    A2,
                    axes=([0, 1], [0, 1]) # temp[alpha',I,beta] A2[alpha',I,beta'] = temp[beta,beta']
            )

        return temp[0, 0]

    @staticmethod
    def l2_distance(mpo1,mpo2):
        """
        Compute the l2 distance between two MPOs: ||rho_1-rho_2||_2.
        """
        if mpo1.n != mpo2.n:
            raise ValueError("The two MPOs must have the same number of qudits.")
        if mpo1.d != mpo2.d:
            raise ValueError("The two MPOs must have the same local dimension.")
        if mpo1.chi != mpo2.chi:
            raise ValueError("The two MPOs must have the same bond dimension.")

        rho1_term = MPOObservables.Inner_product(mpo1,mpo1).real
        rho2_term = MPOObservables.Inner_product(mpo2,mpo2).real
        cross_term = MPOObservables.Inner_product(mpo1,mpo2).real
        return np.sqrt(rho1_term+rho2_term-2*cross_term)
    
    @staticmethod
    def l2_distance_prob(mpo1, mpo2):
        """
        Compute the l2 distance between two output distributions of MPO: ||p_1-p_2||_1.
        """
        if mpo1.n != mpo2.n:
            raise ValueError("The two MPOs must have the same number of qudits.")
        if mpo1.d != mpo2.d:
            raise ValueError("The two MPOs must have the same local dimension.")
        if mpo1.chi != mpo2.chi:
            raise ValueError("The two MPOs must have the same bond dimension.")

        p1_term = MPOObservables.Inner_product_prob(mpo1, mpo1).real
        p2_term = MPOObservables.Inner_product_prob(mpo2, mpo2).real
        cross_term = MPOObservables.Inner_product_prob(mpo1, mpo2).real
        return np.sqrt(p1_term + p2_term - 2 * cross_term)

    @staticmethod
    def TVD(mpo1,mpo2, bit_length = 0):
        """
        Compute the total variance distance between two MPOs: ||p_1-p_2||_1.
        If bit_length == 0, 
        Otherwise, it calculates TVD piece by piece with the bits of bit_length.
        """
        if mpo1.n != mpo2.n:
            raise ValueError("The two MPOs must have the same number of qudits.")
        if mpo1.d != mpo2.d:
            raise ValueError("The two MPOs must have the same local dimension.")
        if mpo1.chi != mpo2.chi:
            raise ValueError("The two MPOs must have the same bond dimension.")
        
        # if bit_length > mpo1.n:
        #     raise ValueError("The bit_length must be less than or equal to the number of qubits in the MPO.")

        if bit_length == 0 or bit_length >= mpo1.n:
            return np.linalg.norm(mpo1.getprobs() - mpo2.getprobs(),1)/2
        else:
            tvd = 0
            fixed_bits = range(2 ** (mpo1.n - bit_length))
            for fixed_bit in fixed_bits:
                tvd += np.linalg.norm(mpo1.getprobs_partial(fixed_bit, bit_length) - mpo2.getprobs_partial(fixed_bit, bit_length),1)/2
            return tvd