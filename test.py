import torch
import numpy as np
enemyCellFeatures = torch.zeros(3 * 7)
enemyCellFeatures[0:3] = torch.from_numpy(np.array([1, 2, 3]))

loo = np.array([i for i in range(10)])
idxs = np.arange(5)
print(loo[idxs])