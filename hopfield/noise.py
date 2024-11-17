import numpy as np


def noise(example, noise_level = 0.01):
    """
    Add noise to an example.
    """

    # Copy the example to avoid modifying the original
    example = np.copy(example)

    # Generate random indices
    indices = np.random.choice(range(len(example)), int(noise_level * len(example)), replace=False)

    # Flip the bits at the random indices
    example[indices] = -example[indices]

    return example