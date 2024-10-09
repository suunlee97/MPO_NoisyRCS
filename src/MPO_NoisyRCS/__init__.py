# Import individual components of MPO
from .core import MPO as MPOCore
from .initialization import MPOInitialization
from .updates import MPOUpdates
from .observables import MPOObservables
from .random_circuits import MPORandom_circuits

# Define the complete MPO class by using multiple inheritance
class MPO(MPOCore, MPOInitialization, MPOUpdates, MPOObservables, MPORandom_circuits):
    """
    A complete Matrix Product Operator (MPO) class, combining all the functionalities.

    This class inherits from:
    - MPOCore: The basic skeleton of the MPO.
    - MPOInitialization: Methods for initializing the MPO.
    - MPOUpdates: Methods for moving the orthogonality center and updating the MPO with quantum gates.
    - MPOObservables: Methods for calculating observables like purity, entanglement entropy, etc.
    - MPORandom_circuits: Methods for generating random gates and noise channels.

    This class allows users to:
    - Initialize an MPO.
    - Update MPO tensors with single-qubit and two-qubit operations.
    - Move the orthogonality center as needed.
    - Calculate observables, such as total probability and entanglement entropy.
    - Generate random quantum gates and apply noise models.
    """

    def __init__(self, n, d, chi, qd, qad, seed=None):
        """
        Initialize the complete MPO class.

        Parameters:
        n (int): Number of qudits.
        d (int): Local dimensionality.
        chi (int): Bond dimension.
        qd (float): Depolarizing probability.
        qad (float): Amplitude damping probability.
        seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(n, d, chi, qd, qad, seed)

    def __str__(self):
        """
        Provide a string representation of the MPO for easy inspection.
        """
        return f"MPO with {self.n} qudits, local dimension {self.d}, bond dimension {self.chi}."

    def display_summary(self):
        """
        Display a summary of the MPO properties.
        """
        print(f"Number of qudits: {self.n}")
        print(f"Local dimension: {self.d}")
        print(f"Bond dimension: {self.chi}")
        print(f"Depolarizing probability: {self.qd}")
        print(f"Amplitude damping probability: {self.qad}")