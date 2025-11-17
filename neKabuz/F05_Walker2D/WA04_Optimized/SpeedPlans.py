import numpy as np

TRAINING_PLANS = [
    # Abiadura bakarra
    np.array([-1.0], dtype = np.float32),
	np.array([-0.5], dtype = np.float32),
	np.array([0.0], dtype = np.float32),
	np.array([0.5], dtype = np.float32),
	np.array([1.0], dtype = np.float32),
	np.array([1.5], dtype = np.float32),
	np.array([2.0], dtype = np.float32),
	np.array([2.5], dtype = np.float32),
	np.array([3.0], dtype = np.float32),

    # Ibili geatu ibili
    np.array([-1.0, -1.0, 0.0, 0.0, -1.0, -1.0], dtype = np.float32),
    np.array([-0.5, -0.5, 0.0, 0.0, -0.5, -0.5], dtype = np.float32),
    np.array([0.5, 0.5, 0.0, 0.0, 0.5, 0.5], dtype = np.float32),
    np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype = np.float32),
    np.array([1.5, 1.5, 0.0, 0.0, 1.5, 1.5], dtype = np.float32),
    np.array([2.0, 2.0, 0.0, 0.0, 2.0, 2.0], dtype = np.float32),
    np.array([2.5, 2.5, 0.0, 0.0, 2.5, 2.5], dtype = np.float32),
    np.array([3.0, 3.0, 0.0, 0.0, 3.0, 3.0], dtype = np.float32),

    # Azeleratu ta frenatu
    np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype = np.float32),
    np.array([0.0, 0.0, -0.5, -0.5, 0.0, 0.0], dtype = np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype = np.float32),
    np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0], dtype = np.float32),
    np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype = np.float32),
    np.array([0.0, 0.0, 1.5, 1.5, 0.0, 0.0], dtype = np.float32),
    np.array([0.0, 0.0, 2.0, 2.0, 0.0, 0.0], dtype = np.float32),
    np.array([0.0, 0.0, 2.5, 2.5, 0.0, 0.0], dtype = np.float32),
    np.array([0.0, 0.0, 3.0, 3.0, 0.0, 0.0], dtype = np.float32),

    # Goraka ta beheraka
    np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype = np.float32),
    np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0], dtype = np.float32),
]




