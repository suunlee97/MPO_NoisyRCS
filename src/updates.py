import numpy as np

class MPOUpdates:
    def MPOMoveOC(self, direction):
        """
        Move the orthogonality center (OC) either left or right.
        
        Parameters:
        direction (str): 'left' or 'right' to specify the direction of movement.
        
        Given that we have the SVD form of A[:, :, :, l] * Lambda[:] * A[:, :, :, l+1],

        left:
        Canonicalize the next left side tensors: A[:, :, :, l-1] * Lambda[:, ] * A[:, :, :, l].

        right:
        Canonicalize the next right side tensors: A[:, :, :, l+1] * Lambda[:, ] * A[:, :, :, l+2].
        Note: l = 0 is automatically done when doing l = 1
        """
        l = self.OC

        if direction == 'left':
            
            if not 1 <= l <= self.n-2:
                raise Exception('Cannot move the orthogonality center to the left further')
            
            # B[alphal-1, Il, alphal]
            B = np.tensordot(self.A[:,:,:,l],
                             np.diag(self.Lambda[:]),
                             axes = ((2), (0)))
            
            Bp = B.reshape(self.chi, self.d ** 2 * self.chi) # (alphal-1, chi * Il + alphal)

            V, Lambda, W = np.linalg.svd(Bp, full_matrices = False) # (chi, chi), (chi), (chi, chi * d ** 2), W[beta, chi * Il + alphal]
            
            AOut1 = np.tensordot(self.A[:,:,:,l-1], V, axes = ((2), (0)))

            AOut2 = np.reshape(W, (self.chi, self.d ** 2, self.chi)) # (beta, Il, alphal)

            self.A[:,:,:,l-1] = AOut1
            self.Lambda[:] = Lambda
            self.A[:,:,:,l] = AOut2

            self.OC = l - 1

        if direction == 'right':
            
            if not 0 <= l <= self.n-3:
                raise Exception('Cannot move the orthogonality center to the right further')
            
            # B[alphal, Il, alphal+1]
            B = np.tensordot(np.diag(self.Lambda[:]),
                             self.A[:,:,:,l+1],
                             axes = ((0), (0)))
            
            Bp = B.reshape(self.chi * self.d ** 2, self.chi)

            V, Lambda, W = np.linalg.svd(Bp, full_matrices = False) # (chi * d ** 2, chi), (chi), (chi, chi), V[d ** 2 * alphal+1 + Il, beta]
            
            AOut1 = np.reshape(V, (self.chi, self.d ** 2, self.chi)) # (alpha, Il, alphal+1)
            
            AOut2 = np.tensordot(W, self.A[:,:,:,l+2], axes = ((1), (0))) # (alpha, beta, Il)

            self.A[:,:,:,l+1] = AOut1
            self.Lambda[:] = Lambda
            self.A[:,:,:,l+2] = AOut2

            self.OC = l + 1

    def MPOtwoqubitUpdate(self, l, GateMPO):

        """
        Update MPO with a two-qubit gate at sites (l, l+1).
        l (int): Cut index where the two-qubit gate is applied. Must satisfy 0 <= l <= n-2.
        GateMPO (np.ndarray): The two-qubit gate in Matrix Product Operator (MPO) form.
        Raises:
        Exception: If the site index 'l' is not in the valid range [0, n-2].
        Exception: If the orthogonality center (OC) is not at the correct site after moving.
        This method updates the MPO tensors A[:,:,:,l], A[:,:,:,l+1] and the singular values Lambda[:]
        by applying the given two-qubit gate in MPO form. It ensures that the orthogonality center (OC)
        is moved to the correct site before performing the update.
        """
        if not 0 <= l <= self.n-2:
            raise Exception('l should be in between 0 and n-2')
        
        # Bring OC to l
        if self.OC < l:
            for i in range(self.OC, l):
                self.MPOMoveOC('right')
        elif self.OC > l:
            for i in range(self.OC, l, -1):
                self.MPOMoveOC('left')
        
        if not self.OC == l:
            raise Exception('OC is not at the correct site')
        
        # Update A[:,:,:,l], A[:,:,:,l+1] and Lambda[:]
        B = np.tensordot(np.tensordot(self.A[:,:,:,l],
                                      np.diag(self.Lambda[:]),
                                      axes = ((2), (0))),
                         self.A[:,:,:,l+1], axes = ((2), (0)))
        
        B = np.tensordot(GateMPO, B, axes = ((2, 3), (1, 2))) # B[Il, Jl, alpha, beta]
        B = np.transpose(B, (2, 0, 1, 3)) # B[alpha, Il, Jl, beta]
        Bp = B.reshape(self.d ** 2 * self.chi, self.d ** 2 * self.chi)

        V, Lambda, W = np.linalg.svd(Bp, full_matrices = False) # V[alpha, beta], Lambda[alpha], W[Il * d ** 2 + Jl, beta]

        V = V.reshape(self.chi, self.d ** 2, self.d ** 2 * self.chi) # V[alpha, Il, Jl]
        W = W.reshape(self.d ** 2 * self.chi, self.d ** 2, self.chi) # W[alpha, Il, Jl]
        AOut1 = V[:, :, :self.chi]
        AOut2 = W[:self.chi, :, :]
        Lambda = Lambda[:self.chi]

        self.A[:,:,:,l] = AOut1
        self.Lambda[:] = Lambda
        self.A[:,:,:,l+1] = AOut2

    def MPOsinglequbitUpdate(self, l, GateMPO):
        """
        Update MPO with a single-qubit gate at site l.
        l (int): Site index where the single-qubit gate is applied. Must satisfy 0 <= l <= n-1.
        GateMPO (np.ndarray): The single-qubit gate in Matrix Product Operator (MPO) form.
        Raises:
        Exception: If the site index 'l' is not in the valid range [0, n-1].
        Exception: If the orthogonality center (OC) is not at the correct site after moving.
        This method updates the MPO tensor A[:,:,:,l] by applying the given single-qubit gate in MPO form.
        It ensures that the orthogonality center (OC) is moved to the correct site before performing the update.
        """
        if not 0 <= l <= self.n-1:
            raise Exception('l should be in between 0 and n-1')
        
        # Bring OC to l or l-1, depending on which one is closer
        if abs(self.OC - l) <= abs(self.OC - (l - 1)):
            target = l
        else:
            target = l - 1

        if self.OC < target:
            for i in range(self.OC, target):
                self.MPOMoveOC('right')
        elif self.OC > target:
            for i in range(self.OC, target, -1):
                self.MPOMoveOC('left')
        if not self.OC == target:
            raise Exception('OC is not at the correct site')
        
        B = np.tensordot(GateMPO,
                         self.A[:,:,:,l],
                         axes = ((1), (1))) # B[Il, alpha, alpha]
        B = np.transpose(B, (1, 0, 2)) # B[alpha, Il, alpha]

        if target == l:
            B = np.tensordot(B,
                             np.diag(self.Lambda),
                             axes = ((2), (0))) # B[alpha, Il, beta]
            Bp = B.reshape(self.chi * self.d ** 2, self.chi)

            V, Lambda, W = np.linalg.svd(Bp, full_matrices = False) # V[alpha, beta], Lambda[alpha], W[Il * d + Jl, beta]
            
            AOut1 = np.reshape(V, (self.chi, self.d ** 2, self.chi)) # V[alpha, Il, Jl]

            AOut2 = np.tensordot(W, self.A[:,:,:,l+1], axes = ((1), (0))) # W[alphal, beta] A[beta, Il, alphal+1]

            self.A[:,:,:,l] = AOut1
            self.Lambda[:] = Lambda
            self.A[:,:,:,l+1] = AOut2
        
        if target == l - 1:
            B = np.tensordot(np.diag(self.Lambda),
                             B,
                             axes = ((1), (0)))
            Bp = B.reshape(self.chi, self.d ** 2 * self.chi)

            U, Lambda, V = np.linalg.svd(Bp, full_matrices = False) # U[alpha, beta], Lambda[alpha], V[Il * d + Jl, beta]

            AOut1 = np.tensordot(self.A[:,:,:,l-1], U, axes = ((2), (0)))

            AOut2 = np.reshape(V, (self.chi, self.d ** 2, self.chi))

            self.A[:,:,:,l-1] = AOut1
            self.Lambda[:] = Lambda
            self.A[:,:,:,l] = AOut2