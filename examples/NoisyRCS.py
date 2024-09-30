# We provide an example of using the MPO class to simulate a noisy random circuit. The circuit is initialized with a product state and then evolved with random gates. The collision probability is calculated at the end of the circuit. The code snippet is as follows:

from src import MPO

import matplotlib.pyplot as plt

# Parameters
n = 20  # Number of qudits
d = 2  # Local dimension (e.g., qubits)
chi = 128  # Bond dimension
qd = 0.1  # Depolarizing probability
qad = 0.05  # Amplitude damping probability
seed = 1  # Random seed for random gates

D = 10  # Depth of the random circuit

# Initialize the MPO
mpo = MPO(n, d, chi, qd, qad, seed)

# Display a summary of the MPO
mpo.display_summary()

# Initialize the MPO with a product state
mpo.MPOInitialization(max_mixed=False)

# Run a random circuit
for t in range(D):
    mpo.RCS1DOneCycleUpdate(t)

probs = mpo.getprobs()
probs.sort()
plt.plot(probs[::-1])

# Calculate collision probability from the MPO
coll_prob = mpo.CollisionProb()
print("Collision Probability:", coll_prob)